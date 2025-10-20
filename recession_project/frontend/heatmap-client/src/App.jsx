import React, { useCallback, useEffect, useMemo, useRef, useState, Suspense, lazy } from 'react'
import PropTypes from 'prop-types'

// Lazy-load Plotly (heavy) and react-plotly factory so initial bundle is small
const Plot = lazy(async () => {
  const plotlyMod = await import('plotly.js-dist-min')
  const Plotly = plotlyMod && plotlyMod.default ? plotlyMod.default : plotlyMod
  const factoryMod = await import('react-plotly.js/factory')
  const createPlotlyComponent = factoryMod && factoryMod.default ? factoryMod.default : factoryMod
  return { default: createPlotlyComponent(Plotly) }
})
import FrameControls from './components/FrameControls.jsx'
import { fetchMeta, fetchFrameRange, fetchTile } from './lib/api.js'
import { idbGet, idbPut, idbHas } from './lib/idb.js'
import './App.css'
const PREFETCH_AHEAD = 8
const DEFAULT_MAX_CACHE_FRAMES = 60

function App({ apiBase }) {
  const [meta, setMeta] = useState(null)
  const [order, setOrder] = useState('clustered')
  const [error, setError] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(300)
  const [cacheVersion, setCacheVersion] = useState(0)

  const frameCacheRef = useRef(new Map())
  const inflightRef = useRef(new Set())
  const currentIndexRef = useRef(0)
  const maxCacheFramesRef = useRef(DEFAULT_MAX_CACHE_FRAMES)
  const [isBulkLoading, setIsBulkLoading] = useState(false)
  const [bulkProgress, setBulkProgress] = useState({ done: 0, total: 0 })
  const progressiveCancelRef = useRef(false)

  useEffect(() => {
    currentIndexRef.current = currentIndex
  }, [currentIndex])

  useEffect(() => {
    // Read order from URL (?order=clustered|diff)
    const params = new URLSearchParams(window.location.search)
    const o = params.get('order')
    if (o === 'diff' || o === 'clustered') {
      setOrder(o)
    }
  }, [])

  useEffect(() => {
    let isMounted = true

    async function loadMeta() {
      console.log('Loading metadata from:', apiBase)
      try {
        const metadata = await fetchMeta(apiBase, order)
        console.log('Metadata loaded:', metadata)
        if (!isMounted) return
        setMeta(metadata)
        setError(null)
      } catch (err) {
        console.error('Failed to load heatmap metadata', err)
        if (!isMounted) return
        const message = err instanceof Error ? err.message : 'Failed to load metadata'
        setError(new Error(message))
      } finally {
        if (isMounted) {
          setIsLoading(false)
        }
      }
    }

  loadMeta()

    return () => {
      isMounted = false
    }
  }, [apiBase, order])

  const trimCache = useCallback(() => {
    if (!meta) return
    const cache = frameCacheRef.current
    if (cache.size <= maxCacheFramesRef.current) {
      return
    }

    const focus = currentIndexRef.current
    const lowerBound = Math.max(0, focus - PREFETCH_AHEAD * 2)
    const upperBound = Math.min(meta.n_dates, focus + PREFETCH_AHEAD * 2)

    for (const key of Array.from(cache.keys())) {
      if (cache.size <= maxCacheFramesRef.current) break
      if (key < lowerBound || key > upperBound) {
        cache.delete(key)
      }
    }
  }, [meta])

  const ensureRange = useCallback(
    async (start, end) => {
      if (!meta) return

      const boundedStart = Math.max(0, start)
      const boundedEnd = Math.min(meta.n_dates, end)
      if (boundedEnd <= boundedStart) return

      const cache = frameCacheRef.current
      let needsFetch = false
      for (let idx = boundedStart; idx < boundedEnd; idx += 1) {
        if (!cache.has(idx)) {
          needsFetch = true
          break
        }
      }
      if (!needsFetch) return

      const inflightKey = `${boundedStart}:${boundedEnd}`
      if (inflightRef.current.has(inflightKey)) {
        return
      }

      inflightRef.current.add(inflightKey)
      try {
  const batch = await fetchFrameRange(apiBase, boundedStart, boundedEnd, order)
        batch.frames.forEach((matrix, offset) => {
          const idx = batch.start + offset
          cache.set(idx, matrix)
          // persist to IndexedDB asynchronously (don't await to keep playback snappy)
          try {
            idbPut(idx, matrix).catch((e) => console.warn('idb put failed', e))
          } catch (e) {
            // ignore
          }
        })
        trimCache()
        setCacheVersion((value) => value + 1)
      } catch (err) {
        console.error('Failed to load heatmap frames', err)
        const message = err instanceof Error ? err.message : 'Failed to load frames'
        setError(new Error(message))
      } finally {
        inflightRef.current.delete(inflightKey)
      }
    },
    [apiBase, meta, trimCache],
  )

  const ensureFrame = useCallback(
    async (index) => {
      if (!meta) return
      if (frameCacheRef.current.has(index)) return
      // try to load from IndexedDB before fetching from network
      try {
        // eslint-disable-next-line no-await-in-loop
        const persisted = await idbGet(index)
        if (persisted !== undefined) {
          frameCacheRef.current.set(index, persisted)
          setCacheVersion((v) => v + 1)
          return
        }
      } catch (e) {
        // ignore idb errors and fallback to network
      }
      await ensureRange(index, index + 1)
    },
    [meta, ensureRange],
  )

  // Bulk loader: fetch frames in batches until all frames fetched.
  const loadAllFrames = useCallback(async (batchSize = 256, delayMs = 50) => {
    if (!meta) return
    if (isBulkLoading) return
    setIsBulkLoading(true)
    // allow cache to grow to full dataset
    maxCacheFramesRef.current = meta.n_dates
    const total = meta.n_dates
    setBulkProgress({ done: 0, total })

    try {
      for (let start = 0; start < total; start += batchSize) {
        const end = Math.min(total, start + batchSize)
        // ensureRange will ignore already cached frames
        // eslint-disable-next-line no-await-in-loop
        await ensureRange(start, end)
        setBulkProgress({ done: Math.min(end, total), total })
        // small pause to avoid saturating CPU/net
        // eslint-disable-next-line no-await-in-loop
        await new Promise((r) => setTimeout(r, delayMs))
      }
    } finally {
      setIsBulkLoading(false)
    }
  }, [meta, ensureRange, isBulkLoading])

  // Progressive tile loader: coarse -> fine levels
  const progressiveLoad = useCallback(async ({ maxLevel = 4, tileSize = 256, onProgress = () => {} } = {}) => {
    if (!meta) return
    if (isBulkLoading) return
    setIsBulkLoading(true)
    maxCacheFramesRef.current = meta.n_dates
    const total = meta.n_dates
    let done = 0

    try {
      // iterate levels from coarse (maxLevel) down to 0
      for (let level = maxLevel; level >= 0; level -= 1) {
        const step = 1 << level
        const framesPerTile = tileSize
        const effectiveTileCount = Math.ceil(total / (framesPerTile * step))
        for (let tile = 0; tile < effectiveTileCount; tile += 1) {
          if (progressiveCancelRef.current) {
            console.info('Progressive load cancelled')
            return
          }
          // check if lower-level already filled these indices to avoid redundant work
          // request tile
          // eslint-disable-next-line no-await-in-loop
          const tileBatch = await fetchTile(apiBase, level, tile, tileSize)
          // tileBatch.frames correspond to sampled frames; map back to original indices
          for (let i = 0; i < tileBatch.frames.length; i += 1) {
            const originalIndex = tileBatch.start + i * tileBatch.step
            // if a finer level already provided this frame, skip overwrite
            if (frameCacheRef.current.has(originalIndex)) {
              // but ensure persisted too
              if (!(await idbHas(originalIndex))) {
                // store persisted copy
                // eslint-disable-next-line no-await-in-loop
                await idbPut(originalIndex, tileBatch.frames[i])
              }
              continue
            }
            frameCacheRef.current.set(originalIndex, tileBatch.frames[i])
            // persist to idb
            // eslint-disable-next-line no-await-in-loop
            await idbPut(originalIndex, tileBatch.frames[i])
            done += 1
          }
          // notify progress
          setBulkProgress({ done: Math.min(done, total), total })
          onProgress({ level, tile, done, total })
        }
      }
    } catch (err) {
      console.error('Progressive load failed', err)
    } finally {
      setCacheVersion((v) => v + 1)
      setIsBulkLoading(false)
    }
  }, [apiBase, meta, isBulkLoading])

  const handleStartProgressive = useCallback(() => {
    progressiveCancelRef.current = false
    progressiveLoad({ maxLevel: 4, tileSize: 256, onProgress: () => {} })
  }, [progressiveLoad])

  const handleCancelBulk = useCallback(() => {
    progressiveCancelRef.current = true
    setIsBulkLoading(false)
  }, [])

  useEffect(() => {
    if (!meta) return
    ensureRange(0, Math.min(meta.n_dates, PREFETCH_AHEAD))
  }, [meta, ensureRange])

  useEffect(() => {
    if (!meta) return
    ensureFrame(currentIndex)
    ensureRange(currentIndex + 1, currentIndex + 1 + PREFETCH_AHEAD)
  }, [currentIndex, meta, ensureFrame, ensureRange])

  useEffect(() => {
    if (!meta || !isPlaying) return undefined

    const timer = setInterval(() => {
      setCurrentIndex((prev) => {
        const next = prev + 1
        if (next >= meta.n_dates) {
          setIsPlaying(false)
          return prev
        }
        return next
      })
    }, playbackSpeed)

    return () => clearInterval(timer)
  }, [isPlaying, playbackSpeed, meta])

  const currentFrame = useMemo(() => {
    if (!meta) return null
    return frameCacheRef.current.get(currentIndex) || null
  }, [cacheVersion, currentIndex, meta])

  const handleIndexChange = useCallback((nextIndex) => {
    setIsPlaying(false)
    setCurrentIndex(nextIndex)
  }, [])

  const togglePlayback = useCallback(() => {
    if (!meta) return
    setIsPlaying((prev) => !prev)
  }, [meta])

  const handleSpeedChange = useCallback((speedMs) => {
    setPlaybackSpeed(speedMs)
  }, [])

  const layout = useMemo(() => {
    if (!meta) return {}
    
    // Generate rainbow spectrum colorscale (Red → Orange → Yellow → Green → Cyan → Blue → Violet)
    // with steps of 0.01 from 0 to 1
    const colorscale = []
    for (let i = 0; i <= 100; i++) {
      const t = i / 100
      let r, g, b
      
      if (t < 0.167) { // Red to Orange (0 to 0.167)
        const local_t = t / 0.167
        r = 255
        g = Math.round(165 * local_t)
        b = 0
      } else if (t < 0.333) { // Orange to Yellow (0.167 to 0.333)
        const local_t = (t - 0.167) / 0.166
        r = 255
        g = Math.round(165 + 90 * local_t)
        b = 0
      } else if (t < 0.5) { // Yellow to Green (0.333 to 0.5)
        const local_t = (t - 0.333) / 0.167
        r = Math.round(255 * (1 - local_t))
        g = 255
        b = 0
      } else if (t < 0.667) { // Green to Cyan (0.5 to 0.667)
        const local_t = (t - 0.5) / 0.167
        r = 0
        g = 255
        b = Math.round(255 * local_t)
      } else if (t < 0.833) { // Cyan to Blue (0.667 to 0.833)
        const local_t = (t - 0.667) / 0.166
        r = 0
        g = Math.round(255 * (1 - local_t))
        b = 255
      } else { // Blue to Violet (0.833 to 1)
        const local_t = (t - 0.833) / 0.167
        r = Math.round(138 * local_t)
        g = 0
        b = 255
      }
      
      colorscale.push([t, `rgb(${r},${g},${b})`])
    }
    
    // Generate gridlines around each cell (Excel-style)
    const n = meta.spreads?.length || 0
    const shapes = []
    
    // Vertical lines (between columns)
    for (let i = 0; i <= n; i++) {
      shapes.push({
        type: 'line',
        x0: i - 0.5,
        x1: i - 0.5,
        y0: -0.5,
        y1: n - 0.5,
        line: {
          color: 'black',
          width: 1
        },
        xref: 'x',
        yref: 'y'
      })
    }
    
    // Horizontal lines (between rows)
    for (let i = 0; i <= n; i++) {
      shapes.push({
        type: 'line',
        x0: -0.5,
        x1: n - 0.5,
        y0: i - 0.5,
        y1: i - 0.5,
        line: {
          color: 'black',
          width: 1
        },
        xref: 'x',
        yref: 'y'
      })
    }
    
    return {
      title: meta.dates?.[currentIndex] || 'Correlation heatmap',
      autosize: true,
      margin: { l: 80, r: 30, t: 60, b: 80 },
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      xaxis: {
        title: 'Spread',
        tickmode: 'array',
        tickvals: Array.from({ length: n }, (_, i) => i),
        ticktext: meta.spreads,
        automargin: true,
        showgrid: false,  // Hide default grid, use shapes instead
        zeroline: false,
      },
      yaxis: {
        title: 'Spread',
        tickmode: 'array',
        tickvals: Array.from({ length: n }, (_, i) => i),
        ticktext: meta.spreads,
        automargin: true,
        showgrid: false,  // Hide default grid, use shapes instead
        zeroline: false,
      },
      shapes: shapes,  // Add Excel-style gridlines
      coloraxis: {
        colorscale: colorscale,
        cmin: 0,
        cmax: 1,
        colorbar: {
          title: 'Correlation',
          titleside: 'right',
          tickmode: 'linear',
          tick0: 0,
          dtick: 0.1,
        },
      },
    }
  }, [meta, currentIndex])

  const plotData = useMemo(() => {
    if (!meta || !currentFrame) return []
    
    // Debug: Check dimensions
    console.log('plotData debug:', {
      frameShape: `${currentFrame.length} x ${currentFrame[0]?.length || 0}`,
      xLabels: meta.spreads.length,
      yLabels: meta.spreads.length,
      firstRow: currentFrame[0]?.length,
      totalRows: currentFrame.length
    })
    
    return [
      {
        z: currentFrame,
        x: meta.spreads,
        y: meta.spreads,
        type: 'heatmap',
        coloraxis: 'coloraxis',
        xgap: 0,  // No gap needed, using shapes for gridlines
        ygap: 0,  // No gap needed, using shapes for gridlines
        hovertemplate: 'Spread X: %{x}<br>Spread Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>',
      },
    ]
  }, [meta, currentFrame])

  return (
    <div className="app-shell">
      <div className="app-content">
        <div className="app-header">
          <h2>Correlation Heatmap Explorer</h2>
          <p>
            Stream frames directly from the FastAPI backend without loading multi-gigabyte HTML. Use the
            playback controls to scrub through the timeline.
          </p>
        </div>

        {error && (
          <div className="app-alert app-alert--error">
            <strong>Something went wrong:</strong> {error.message}
          </div>
        )}

        {isLoading && (
          <div className="app-loader">Loading metadata...</div>
        )}

        {!isLoading && !error && meta && (
          <>
            <FrameControls
              currentIndex={currentIndex}
              dateLabel={meta.dates?.[currentIndex] || '-'}
              totalFrames={meta.n_dates}
              isPlaying={isPlaying}
              playbackSpeed={playbackSpeed}
              onIndexChange={handleIndexChange}
              onTogglePlay={togglePlayback}
              onSpeedChange={handleSpeedChange}
              onLoadAll={() => loadAllFrames(256, 25)}
              isBulkLoading={isBulkLoading}
              bulkProgress={bulkProgress}
              onCancelBulk={handleCancelBulk}
              onStartProgressive={handleStartProgressive}
            />

            <div className="plot-wrapper">
              {currentFrame ? (
                <Suspense fallback={<div className="app-loader">Loading plot library...</div>}>
                  <Plot
                    data={plotData}
                    layout={layout}
                    config={{ responsive: true, displaylogo: false }}
                    useResizeHandler
                    style={{ width: '100%', height: '100%' }}
                  />
                </Suspense>
              ) : (
                <div className="app-loader">Fetching frame...</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

App.propTypes = {
  apiBase: PropTypes.string,
}

App.defaultProps = {
  apiBase: '',
}

export default App
