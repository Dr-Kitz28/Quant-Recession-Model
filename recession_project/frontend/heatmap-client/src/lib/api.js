const DEFAULT_HEADERS = {
  Accept: "application/json",
}

function sanitizeBase(baseUrl) {
  if (!baseUrl) return ''
  return baseUrl.replace(/\/$/, '')
}

function raiseForStatus(response, context) {
  if (response.ok) return
  const error = new Error(`${context} failed with status ${response.status}: ${response.statusText || 'Unknown error'}`)
  error.status = response.status
  error.statusText = response.statusText
  throw error
}

export async function fetchMeta(baseUrl, order = 'clustered') {
  const url = `${sanitizeBase(baseUrl)}/meta?order=${encodeURIComponent(order)}`
  const response = await fetch(url, {
    headers: DEFAULT_HEADERS,
  })
  raiseForStatus(response, 'Fetching metadata')
  return response.json()
}

export async function fetchFrameRange(baseUrl, start, end, order = 'clustered') {
  const params = new URLSearchParams({ start: String(start), end: String(end), order: String(order) })
  const url = `${sanitizeBase(baseUrl)}/frames?${params.toString()}`
  const response = await fetch(url)
  raiseForStatus(response, 'Fetching frame range')

  const rows = Number(response.headers.get('X-Rows')) || 0
  const cols = Number(response.headers.get('X-Cols')) || 0
  const batchStart = Number(response.headers.get('X-Start')) || start
  const batchEnd = Number(response.headers.get('X-End')) || end

  if (!rows || !cols) {
    throw new Error('Frame dimensions missing from response headers')
  }

  const buffer = await response.arrayBuffer()
  const values = new Float32Array(buffer)
  const frameSize = rows * cols
  const frameCount = values.length / frameSize

  if (!Number.isInteger(frameCount)) {
    throw new Error('Frame payload size does not align with declared dimensions')
  }

  const frames = []
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const startOffset = frameIndex * frameSize
    const slice = values.subarray(startOffset, startOffset + frameSize)
    const matrix = []
    for (let row = 0; row < rows; row += 1) {
      const rowStart = row * cols
      const rowSlice = slice.subarray(rowStart, rowStart + cols)
      matrix.push(Array.from(rowSlice))
    }
    frames.push(matrix)
  }

  return {
    start: batchStart,
    end: batchEnd,
    frames,
  }
}

export async function fetchTile(baseUrl, level = 0, tile = 0, tile_size = 256) {
  const params = new URLSearchParams({ level: String(level), tile: String(tile), tile_size: String(tile_size) })
  const url = `${sanitizeBase(baseUrl)}/tiles?${params.toString()}`
  const response = await fetch(url, {
    headers: DEFAULT_HEADERS,
  })
  raiseForStatus(response, 'Fetching tile')

  const rows = Number(response.headers.get('X-Rows')) || 0
  const cols = Number(response.headers.get('X-Cols')) || 0
  const start = Number(response.headers.get('X-Start')) || 0
  const end = Number(response.headers.get('X-End')) || 0
  const step = Number(response.headers.get('X-Step')) || 1
  const count = Number(response.headers.get('X-Count')) || 0

  if (!rows || !cols) {
    throw new Error('Frame dimensions missing from tile response headers')
  }

  const buffer = await response.arrayBuffer()
  const values = new Float32Array(buffer)
  const frameSize = rows * cols
  const frameCount = values.length / frameSize

  if (!Number.isInteger(frameCount)) {
    throw new Error('Tile payload size does not align with declared dimensions')
  }

  const frames = []
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const startOffset = frameIndex * frameSize
    const slice = values.subarray(startOffset, startOffset + frameSize)
    const matrix = []
    for (let row = 0; row < rows; row += 1) {
      const rowStart = row * cols
      const rowSlice = slice.subarray(rowStart, rowStart + cols)
      matrix.push(Array.from(rowSlice))
    }
    frames.push(matrix)
  }

  return {
    level: Number(level),
    tile: Number(tile),
    start,
    end,
    step,
    count,
    rows,
    cols,
    frames,
  }
}
