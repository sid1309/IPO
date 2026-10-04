import React, { useState, useEffect } from 'react';
import {
  FileText,
  BarChart3,
  MessageSquare,
  Network,
  UploadCloud,
  ChevronDown,
  ShieldCheck,
  CheckCircle,
} from 'lucide-react';
import SummaryTab from './components/SummaryTab';
import ChatTab from './components/ChatTab';
import GraphTab from './components/GraphTab';
import UploadModal from './components/UploadModal';

export default function App() {
  const [ipos, setIpos] = useState([]);
  const [activeIpoId, setActiveIpoId] = useState(
    localStorage.getItem('active_ipo_id') || 'zomato-2021'
  );
  const [activeTab, setActiveTab] = useState('summary');
  const [summaryData, setSummaryData] = useState(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [loadingSummary, setLoadingSummary] = useState(false);

  // 1. Fetch available IPOs
  const fetchIpos = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/ipos');
      if (res.ok) {
        const data = await res.json();
        setIpos(data);
        if (data.length > 0 && !activeIpoId) {
          setActiveIpoId(data[0].ipo_id);
        }
      }
    } catch (e) {
      console.error('Error fetching IPO filings:', e);
    }
  };

  useEffect(() => {
    fetchIpos();
  }, []);

  // 2. Fetch summary card whenever active IPO changes
  useEffect(() => {
    if (!activeIpoId) return;
    localStorage.setItem('active_ipo_id', activeIpoId);

    async function loadSummary() {
      setLoadingSummary(true);
      try {
        const res = await fetch(`http://localhost:8000/api/v1/summary/${activeIpoId}`);
        if (res.ok) {
          const data = await res.json();
          setSummaryData(data);
        } else {
          setSummaryData(null);
        }
      } catch (e) {
        setSummaryData(null);
      } finally {
        setLoadingSummary(false);
      }
    }
    loadSummary();
  }, [activeIpoId]);

  const activeIpo = ipos.find((i) => i.ipo_id === activeIpoId);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon">
            <FileText size={22} />
          </div>
          <div className="brand-text">
            <h1>IPO Prospectus Analyst</h1>
            <p>Institutional Hybrid RAG & GraphRAG Platform</p>
          </div>
        </div>

        <div className="header-actions">
          {/* Active IPO Selector Dropdown */}
          <div className="select-wrapper">
            <select
              className="ipo-select"
              value={activeIpoId}
              onChange={(e) => setActiveIpoId(e.target.value)}
            >
              {ipos.map((ipo) => (
                <option key={ipo.ipo_id} value={ipo.ipo_id}>
                  {ipo.company_name} ({ipo.page_count} Pages)
                </option>
              ))}
            </select>
            <ChevronDown size={16} className="select-arrow" />
          </div>

          {/* Upload Button */}
          <button className="upload-btn" onClick={() => setIsUploadOpen(true)}>
            <UploadCloud size={16} />
            <span>Upload Prospectus</span>
          </button>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="tab-navigation">
        <button
          className={`tab-btn ${activeTab === 'summary' ? 'active' : ''}`}
          onClick={() => setActiveTab('summary')}
        >
          <BarChart3 size={17} />
          <span>Structured Summary & Proceeds</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          <MessageSquare size={17} />
          <span>Prospectus Analyst Q&A</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'graph' ? 'active' : ''}`}
          onClick={() => setActiveTab('graph')}
        >
          <Network size={17} />
          <span>GraphRAG Corporate Network</span>
        </button>
      </nav>

      {/* Main Workspace */}
      <main className="main-content">
        {loadingSummary && activeTab === 'summary' ? (
          <div className="glass-card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem' }}>
            <div className="spin" style={{ width: '28px', height: '28px', border: '3px solid #60A5FA', borderTopColor: 'transparent', borderRadius: '50%', margin: '0 auto 1rem' }} />
            <p style={{ color: 'var(--text-secondary)' }}>Loading verified structured financial card...</p>
          </div>
        ) : (
          <>
            {activeTab === 'summary' && (
              <SummaryTab summary={summaryData} ipoInfo={activeIpo} />
            )}
            {activeTab === 'chat' && (
              <ChatTab ipoId={activeIpoId} companyName={activeIpo?.company_name} />
            )}
            {activeTab === 'graph' && (
              <GraphTab ipoId={activeIpoId} companyName={activeIpo?.company_name} />
            )}
          </>
        )}
      </main>

      {/* SEBI Compliance Regulatory Footer */}
      <footer className="sebi-footer">
        <p>
          ⚖️ <strong>Regulatory Notice (SEBI Compliance)</strong>: This is an educational document-research platform.
          Under Section 12A of the SEBI Act, 1992 and SEBI (Research Analysts) Regulations, 2014, providing investment advice requires formal registration.
          This tool extracts and synthesizes factual regulatory disclosures only and does not recommend to buy, sell, apply, or avoid any security.
        </p>
      </footer>

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadComplete={(newIpoId) => {
          fetchIpos();
          setActiveIpoId(newIpoId);
          setActiveTab('summary');
        }}
      />
    </div>
  );
}
