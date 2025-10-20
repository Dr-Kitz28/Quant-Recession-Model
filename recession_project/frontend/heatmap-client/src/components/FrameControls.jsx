import PropTypes from "prop-types"

const SPEED_OPTIONS = [
  { label: "0.5", value: 600 },
  { label: "1", value: 300 },
  { label: "2", value: 150 },
  { label: "4", value: 75 },
]

function FrameControls({
  currentIndex,
  totalFrames,
  dateLabel,
  isPlaying,
  playbackSpeed,
  onIndexChange,
  onTogglePlay,
  onSpeedChange,
  onLoadAll,
  isBulkLoading,
  bulkProgress,
  onCancelBulk,
}) {
  if (!Number.isFinite(totalFrames) || totalFrames <= 0) {
    return null
  }

  return (
    <div className="frame-controls">
      <div className="frame-controls__row frame-controls__row--top">
        <button type="button" className="frame-controls__button" onClick={onTogglePlay}>
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button type="button" className="frame-controls__button" onClick={() => onLoadAll?.()} style={{ marginLeft: '0.5rem' }} disabled={isBulkLoading}>
          {isBulkLoading ? 'Loading...' : 'Load all frames'}
        </button>
        <button type="button" className="frame-controls__button" onClick={() => onStartProgressive?.()} style={{ marginLeft: '0.5rem' }} disabled={isBulkLoading}>
          Progressive load
        </button>
        {isBulkLoading && (
          <button type="button" className="frame-controls__button" onClick={() => onCancelBulk?.()} style={{ marginLeft: '0.5rem' }}>
            Cancel
          </button>
        )}
        <div className="frame-controls__status">
          <span className="frame-controls__frame">
            Frame {currentIndex + 1} / {totalFrames}
          </span>
          <span className="frame-controls__date">{dateLabel}</span>
        </div>
        <label className="frame-controls__speed">
          Speed
          <select value={playbackSpeed} onChange={(event) => onSpeedChange(Number(event.target.value))}>
            {SPEED_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <input
        type="range"
        min="0"
        max={totalFrames - 1}
        value={currentIndex}
        onChange={(event) => onIndexChange(Number(event.target.value))}
        className="frame-controls__slider"
        aria-label="Frame selector"
      />
      {isBulkLoading && bulkProgress && (
        <div style={{ marginTop: '0.5rem' }}>
          <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 6, overflow: 'hidden' }}>
            <div style={{ width: `${Math.round((bulkProgress.done / bulkProgress.total) * 100)}%`, height: '100%', background: 'var(--accent-color, #03dac6)' }} />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--secondary-text, #9ba8b4)', marginTop: '0.25rem' }}>{bulkProgress.done} / {bulkProgress.total} frames</div>
        </div>
      )}
    </div>
  )
}

FrameControls.propTypes = {
  currentIndex: PropTypes.number.isRequired,
  totalFrames: PropTypes.number.isRequired,
  dateLabel: PropTypes.string,
  isPlaying: PropTypes.bool.isRequired,
  playbackSpeed: PropTypes.number.isRequired,
  onIndexChange: PropTypes.func.isRequired,
  onTogglePlay: PropTypes.func.isRequired,
  onSpeedChange: PropTypes.func.isRequired,
  onLoadAll: PropTypes.func,
  isBulkLoading: PropTypes.bool,
  bulkProgress: PropTypes.object,
  onCancelBulk: PropTypes.func,
  onStartProgressive: PropTypes.func,
}

FrameControls.defaultProps = {
  dateLabel: "–",
  onLoadAll: null,
  isBulkLoading: false,
  bulkProgress: null,
  onCancelBulk: null,
  onStartProgressive: null,
}

export default FrameControls

