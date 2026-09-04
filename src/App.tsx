import { useEffect, useState } from 'react'

type ServiceStatus = 'checking' | 'ready' | 'offline'

export function App() {
  const [status, setStatus] = useState<ServiceStatus>('checking')
  const [videoName, setVideoName] = useState('')

  useEffect(() => {
    fetch('/api/health')
      .then((response) => setStatus(response.ok ? 'ready' : 'offline'))
      .catch(() => setStatus('offline'))
  }, [])

  return (
    <main>
      <header>
        <p className="eyebrow">Local-first volleyball video editing</p>
        <h1>VolleyMole</h1>
        <p className="lede">Dig rallies out of continuous match footage—on your own machine.</p>
      </header>

      <section className="panel" aria-labelledby="new-analysis">
        <div className="panel-heading">
          <div>
            <h2 id="new-analysis">New analysis</h2>
            <p>Choose continuous footage of one clearly visible court.</p>
          </div>
          <span className={`status status-${status}`}>Local service: {status}</span>
        </div>

        <label className="picker">
          <span>{videoName || 'Select a match video'}</span>
          <input
            type="file"
            accept="video/*"
            onChange={(event) => setVideoName(event.target.files?.[0]?.name ?? '')}
          />
        </label>

        <button type="button" disabled={!videoName || status !== 'ready'}>
          Start local analysis
        </button>
        <p className="note">The v1 analysis pipeline is not wired yet. Video data stays local.</p>
      </section>
    </main>
  )
}
