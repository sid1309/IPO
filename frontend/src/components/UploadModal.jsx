import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, CheckCircle2, Loader2, X, FileText, AlertCircle, Clock } from 'lucide-react';

export default function UploadModal({ isOpen, onClose, onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const timerRef = useRef(null);

  if (!isOpen) return null;

  const steps = [
    {
      title: 'Parsing PDF & Extracting Text',
      desc: 'Loading pages via PyMuPDF and extracting structural layout...',
    },
    {
      title: 'Detecting SEBI Prospectus Chapters & Tables',
      desc: 'Scanning chapter boundaries across all pages and isolating financial tables...',
    },
    {
      title: 'Generating Dense + Sparse BM25 Hybrid Embeddings',
      desc: 'Computing BAAI & BM25 vectors on CPU and upserting into Qdrant (~1.5–3 min for 500+ pages)...',
    },
    {
      title: 'Extracting Structured Summary & Objects of Issue',
      desc: 'Extracting price band, OFS/Fresh split, capital allocation, and key risks...',
    },
    {
      title: 'Building Neo4j Corporate Knowledge Graph',
      desc: 'Resolving entities and building multi-hop promoter, subsidiary, and litigation network...',
    },
  ];

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      if (selected.name.toLowerCase().endsWith('.pdf')) {
        setFile(selected);
        setError(null);
      } else {
        setError('Please select a valid PDF prospectus document.');
      }
    }
  };

  const formatTime = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}m ${s < 10 ? '0' : ''}${s}s`;
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setError(null);
    setCurrentStep(0);
    setElapsedSeconds(0);

    // Realistic timer & step pacing for large 400-600 page institutional prospectuses
    let seconds = 0;
    timerRef.current = setInterval(() => {
      seconds += 1;
      setElapsedSeconds(seconds);

      // Pacing calibrated to real backend CPU processing times:
      // 0-12s: Parsing PDF
      // 12-30s: Section Detection & Chunking
      // 30-130s: Hybrid Vector Embedding & Qdrant Upsert
      // 130-170s: Summary Extraction
      // 170s+: Knowledge Graph Generation
      if (seconds < 12) {
        setCurrentStep(0);
      } else if (seconds < 30) {
        setCurrentStep(1);
      } else if (seconds < 130) {
        setCurrentStep(2);
      } else if (seconds < 170) {
        setCurrentStep(3);
      } else {
        setCurrentStep(4);
      }
    }, 1000);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('http://localhost:8000/api/v1/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(timerRef.current);

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Upload and indexing failed.');
      }

      const data = await res.json();
      setCurrentStep(steps.length);
      setTimeout(() => {
        setIsUploading(false);
        onUploadComplete(data.ipo_id);
        onClose();
      }, 1000);
    } catch (err) {
      clearInterval(timerRef.current);
      setIsUploading(false);
      setError(err.message || 'An error occurred during filing ingestion.');
    }
  };

  return (
    <div className="modal-overlay" onClick={!isUploading ? onClose : undefined}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'white' }}>Upload IPO Prospectus</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Ingest DRHP or RHP filing for AI analysis</p>
          </div>
          {!isUploading && (
            <button
              onClick={onClose}
              style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
            >
              <X size={20} />
            </button>
          )}
        </div>

        {!isUploading ? (
          <>
            <div
              className={`drop-zone ${file ? 'active' : ''}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
              <UploadCloud size={44} style={{ color: 'var(--accent-blue)', margin: '0 auto 0.75rem' }} />
              {file ? (
                <div>
                  <p style={{ fontWeight: 600, color: 'white' }}>{file.name}</p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {(file.size / (1024 * 1024)).toFixed(2)} MB • Click to replace
                  </p>
                </div>
              ) : (
                <div>
                  <p style={{ fontWeight: 600, color: 'white' }}>Drop prospectus PDF here or click to browse</p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    Supports DRHP / RHP filings up to 100MB
                  </p>
                </div>
              )}
            </div>

            {error && (
              <div style={{ display: 'flex', gap: '0.5rem', color: '#F87171', fontSize: '0.8rem', marginBottom: '1rem', background: 'rgba(239, 68, 68, 0.08)', padding: '0.65rem', borderRadius: '6px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>{error}</span>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button
                onClick={onClose}
                style={{
                  background: 'none',
                  border: '1px solid var(--border-light)',
                  color: 'var(--text-secondary)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.55rem 1rem',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                }}
              >
                Cancel
              </button>
              <button
                disabled={!file}
                onClick={handleUpload}
                className="upload-btn"
                style={{ opacity: file ? 1 : 0.5 }}
              >
                Start Ingestion
              </button>
            </div>
          </>
        ) : (
          <div style={{ padding: '0.5rem 0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <p style={{ fontSize: '0.9rem', fontWeight: 600, color: 'white' }}>
                Ingesting: <span style={{ color: '#60A5FA' }}>{file?.name}</span>
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem', color: '#93C5FD', background: 'rgba(59, 130, 246, 0.15)', padding: '0.2rem 0.6rem', borderRadius: '9999px', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                <Clock size={13} />
                <span>{formatTime(elapsedSeconds)}</span>
              </div>
            </div>

            <div className="stepper">
              {steps.map((st, idx) => {
                const isDone = currentStep > idx;
                const isActive = currentStep === idx;
                return (
                  <div
                    key={idx}
                    className={`step-item ${isDone ? 'done' : isActive ? 'active' : ''}`}
                    style={{ alignItems: 'flex-start' }}
                  >
                    <div style={{ marginTop: '2px' }}>
                      {isDone ? (
                        <CheckCircle2 size={18} style={{ color: '#34D399', flexShrink: 0 }} />
                      ) : isActive ? (
                        <Loader2 size={18} className="spin" style={{ color: '#60A5FA', flexShrink: 0 }} />
                      ) : (
                        <div
                          style={{
                            width: 18,
                            height: 18,
                            borderRadius: '50%',
                            border: '2px solid #374151',
                            flexShrink: 0,
                          }}
                        />
                      )}
                    </div>
                    <div>
                      <div style={{ fontWeight: isActive ? 600 : 500 }}>{st.title}</div>
                      {isActive && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
                          {st.desc}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ marginTop: '1.25rem', padding: '0.65rem 0.85rem', background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)', borderRadius: '6px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              ℹ️ <strong>Note</strong>: Institutional filings with 500+ pages generate ~2,500 chunks. Dense & sparse BM25 embeddings are calculated locally on CPU for privacy, taking ~1.5–3 minutes.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
