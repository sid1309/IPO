import React, { useState, useEffect } from 'react';
import { Network, AlertOctagon, Users, ArrowRight, BookOpen, Building2, Gavel } from 'lucide-react';

export default function GraphTab({ ipoId, companyName }) {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchGraph() {
      if (!ipoId) return;
      setLoading(true);
      try {
        const res = await fetch(`http://localhost:8000/api/v1/graph/${ipoId}`);
        if (res.ok) {
          const data = await res.json();
          setGraphData(data);
        }
      } catch (e) {
        console.error('Failed to load graph data:', e);
      } finally {
        setLoading(false);
      }
    }
    fetchGraph();
  }, [ipoId]);

  if (loading) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem' }}>
        <div className="spin" style={{ width: '28px', height: '28px', border: '3px solid #60A5FA', borderTopColor: 'transparent', borderRadius: '50%', margin: '0 auto 1rem' }} />
        <p style={{ color: 'var(--text-secondary)' }}>Traversing Neo4j corporate relationship graph...</p>
      </div>
    );
  }

  const litigations = graphData?.promoter_litigations || [];
  const directorships = graphData?.common_directorships || [];
  const nodes = graphData?.nodes || [];
  const relationships = graphData?.relationships || [];

  return (
    <div>
      {/* Overview Cards */}
      <div className="kpi-grid">
        <div className="kpi-card purple">
          <div className="kpi-label">Entity Nodes</div>
          <div className="kpi-value">{nodes.length}</div>
          <div className="kpi-sub">Companies, Promoters, Litigations</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Directed Relationships</div>
          <div className="kpi-value">{relationships.length}</div>
          <div className="kpi-sub">Edges: Director, Promoter, Party_To</div>
        </div>
        <div className="kpi-card amber">
          <div className="kpi-label">Multi-Hop Litigations</div>
          <div className="kpi-value">{litigations.length}</div>
          <div className="kpi-sub">Connected via Group Companies</div>
        </div>
        <div className="kpi-card emerald">
          <div className="kpi-label">Common Directorships</div>
          <div className="kpi-value">{directorships.length}</div>
          <div className="kpi-sub">Cross-Board Memberships</div>
        </div>
      </div>

      {/* Multi-Hop Promoter Litigations Section */}
      <div className="glass-card" style={{ marginBottom: '1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'white' }}>
              Multi-Hop Promoter & Subsidiary Litigations (Graph Traversal)
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Deterministic Cypher path: <code>(Promoter)-[:DIRECTOR_OF]-&gt;(GroupCompany)-[:PARTY_TO]-&gt;(Litigation)</code>
            </p>
          </div>
          <span className="badge badge-amber">
            <AlertOctagon size={13} />
            Multi-Hop Risk Analysis
          </span>
        </div>

        {litigations.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {litigations.map((item, i) => (
              <div
                key={i}
                style={{
                  background: 'linear-gradient(145deg, #18233C, #111A2E)',
                  border: '1px solid rgba(245, 158, 11, 0.25)',
                  borderRadius: 'var(--radius-md)',
                  padding: '1.25rem',
                }}
              >
                {/* Path Visualizer */}
                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                  {/* Step 1: Promoter */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'rgba(99, 102, 241, 0.15)', padding: '0.35rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                    <Users size={15} style={{ color: '#818CF8' }} />
                    <span style={{ fontWeight: 700, color: '#C7D2FE', fontSize: '0.85rem' }}>{item.promoter}</span>
                    <span style={{ fontSize: '0.7rem', color: '#9CA3AF' }}>(Page {item.promoter_page || '—'})</span>
                  </div>

                  <ArrowRight size={16} style={{ color: 'var(--text-muted)' }} />

                  {/* Step 2: Group Company */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'rgba(59, 130, 246, 0.15)', padding: '0.35rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                    <Building2 size={15} style={{ color: '#60A5FA' }} />
                    <span style={{ fontWeight: 700, color: '#BFDBFE', fontSize: '0.85rem' }}>{item.group_company}</span>
                    <span style={{ fontSize: '0.7rem', color: '#9CA3AF' }}>(Page {item.group_page || '—'})</span>
                  </div>

                  <ArrowRight size={16} style={{ color: 'var(--text-muted)' }} />

                  {/* Step 3: Litigation */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'rgba(239, 68, 68, 0.15)', padding: '0.35rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                    <Gavel size={15} style={{ color: '#F87171' }} />
                    <span style={{ fontWeight: 700, color: '#FECACA', fontSize: '0.85rem' }}>{item.litigation}</span>
                    <span style={{ fontSize: '0.7rem', color: '#9CA3AF' }}>(Page {item.litigation_page || '—'})</span>
                  </div>
                </div>

                {/* Details Row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <span>Disputed Amount: <strong style={{ color: '#34D399' }}>{item.litigation_amount}</strong></span>
                  <span>Status: <strong style={{ color: '#FBBF24' }}>{item.litigation_status}</strong></span>
                  <span>Primary Section: <strong style={{ color: '#93C5FD' }}>Outstanding Litigation</strong></span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '2rem 1rem', color: 'var(--text-muted)' }}>
            No multi-hop promoter litigations found in this filing.
          </div>
        )}
      </div>

      {/* Common Directorships */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'white' }}>
              Common Directorships (Board Overlaps)
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Directors serving on both the Issuer Company and Group Companies
            </p>
          </div>
          <span className="badge badge-emerald">Governance Graph</span>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Director Name</th>
              <th>Group Company / Subsidiary</th>
              <th>Document Citation</th>
            </tr>
          </thead>
          <tbody>
            {directorships && directorships.length > 0 ? (
              directorships.map((d, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600, color: 'white' }}>{d.person_name}</td>
                  <td style={{ color: '#93C5FD' }}>{d.group_company}</td>
                  <td>
                    <span className="citation-pill">Page {d.page || 'Board Profile'}</span>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="3" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                  No overlapping board directorships detected.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
