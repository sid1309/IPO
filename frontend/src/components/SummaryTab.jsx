import React from 'react';
import { TrendingUp, DollarSign, Layers, AlertTriangle, ShieldCheck, PieChart } from 'lucide-react';

export default function SummaryTab({ summary, ipoInfo }) {
  if (!summary) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem' }}>
        <PieChart size={48} style={{ color: 'var(--text-muted)', margin: '0 auto 1rem' }} />
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'white' }}>No Summary Card Available</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
          This prospectus has not been indexed yet. Ingestion will extract structured financial metrics.
        </p>
      </div>
    );
  }

  const { issue_details, objects_of_issue, financials, promoters, key_risks_summary } = summary;

  return (
    <div>
      {/* KPI Grid */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Price Band</div>
          <div className="kpi-value">
            {issue_details.price_band_floor && issue_details.price_band_cap
              ? `₹ ${issue_details.price_band_floor} - ₹ ${issue_details.price_band_cap}`
              : 'Book Building'}
          </div>
          <div className="kpi-sub">Face Value: ₹ {issue_details.face_value || '1.00'}</div>
        </div>

        <div className="kpi-card emerald">
          <div className="kpi-label">Total Issue Size</div>
          <div className="kpi-value">{issue_details.total_issue_size || 'N/A'}</div>
          <div className="kpi-sub">
            Fresh: {issue_details.fresh_issue_amount || 'N/A'} • OFS: {issue_details.ofs_amount || 'N/A'}
          </div>
        </div>

        <div className="kpi-card amber">
          <div className="kpi-label">Bid Lot Size</div>
          <div className="kpi-value">{issue_details.lot_size ? `${issue_details.lot_size} Shares` : 'N/A'}</div>
          <div className="kpi-sub">Exchanges: {issue_details.listing_exchanges?.join(', ') || 'BSE, NSE'}</div>
        </div>

        <div className="kpi-card purple">
          <div className="kpi-label">Issue Structure</div>
          <div className="kpi-value">{issue_details.issue_type || '100% Book Built'}</div>
          <div className="kpi-sub">Filing Type: {ipoInfo?.doc_type || 'RHP'}</div>
        </div>
      </div>

      {/* Fresh Issue vs OFS Split Progress */}
      <div className="glass-card" style={{ marginBottom: '1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'white' }}>Capital Dilution & Supply Overhang</h3>
          <span className="badge badge-blue">Offer Breakdown</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginTop: '0.75rem' }}>
          <span style={{ color: '#60A5FA' }}>Fresh Growth Capital: {issue_details.fresh_issue_amount || 'N/A'}</span>
          <span style={{ color: '#FBBF24' }}>Offer For Sale (OFS): {issue_details.ofs_amount || 'N/A'}</span>
        </div>
        <div className="progress-bar-bg">
          <div className="progress-bar-fill" style={{ width: '92%', background: '#3B82F6' }} />
          <div className="progress-bar-fill" style={{ width: '8%', background: '#F59E0B' }} />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.75rem', marginBottom: '1.75rem' }}>
        {/* Objects of the Issue Table */}
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'white' }}>Use of Fresh Issue Proceeds</h3>
            <span className="badge badge-emerald">Itemized Objects</span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Object / Purpose</th>
                <th>Estimated Amount</th>
                <th>Deployment Timeline</th>
              </tr>
            </thead>
            <tbody>
              {objects_of_issue && objects_of_issue.length > 0 ? (
                objects_of_issue.map((obj, i) => (
                  <tr key={i}>
                    <td>
                      <div style={{ fontWeight: 600, color: 'white' }}>{obj.object_category}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{obj.description}</div>
                    </td>
                    <td style={{ fontWeight: 700, color: '#34D399', whiteSpace: 'nowrap' }}>
                      {obj.estimated_amount}
                    </td>
                    <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {obj.deployment_schedule || 'As needed'}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="3" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                    No itemized objects extracted.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Financial Snapshot Table */}
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'white' }}>Key Financial Metrics</h3>
            <span className="badge badge-amber">Historical Performance</span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Period</th>
                <th>Revenue</th>
                <th>EBITDA</th>
                <th>PAT (Loss)</th>
                <th>EPS</th>
                <th>RoNW</th>
              </tr>
            </thead>
            <tbody>
              {financials && financials.length > 0 ? (
                financials.map((f, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 600, color: 'white' }}>{f.fiscal_year}</td>
                    <td>{f.revenue || '—'}</td>
                    <td>{f.ebitda || '—'}</td>
                    <td style={{ color: f.pat?.includes('(') ? '#F87171' : '#34D399', fontWeight: 600 }}>
                      {f.pat || '—'}
                    </td>
                    <td>{f.diluted_eps || '—'}</td>
                    <td>{f.ronw || '—'}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="6" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                    No financial history extracted.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Promoters and Risk Factors */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.75rem' }}>
        {/* Promoters Table */}
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'white' }}>Promoters & Selling Shareholders</h3>
            <span className="badge badge-purple">Shareholding</span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Entity / Promoter Name</th>
                <th>Pre-Offer %</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {promoters && promoters.length > 0 ? (
                promoters.map((p, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 600, color: 'white' }}>{p.name}</td>
                    <td style={{ fontWeight: 700, color: '#60A5FA' }}>{p.pre_offer_percentage || '—'}</td>
                    <td>
                      <span className={`badge ${p.is_selling_shareholder ? 'badge-amber' : 'badge-emerald'}`}>
                        {p.is_selling_shareholder ? 'Selling Shareholder (OFS)' : 'Promoter'}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="3" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                    No promoter data extracted.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Key Risk Factors */}
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'white' }}>Disclosed Key Risk Factors</h3>
            <span className="badge badge-amber">Section: Risk Factors</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {key_risks_summary && key_risks_summary.length > 0 ? (
              key_risks_summary.map((risk, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    gap: '0.65rem',
                    background: 'rgba(239, 68, 68, 0.05)',
                    border: '1px solid rgba(239, 68, 68, 0.2)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '0.75rem',
                    fontSize: '0.82rem',
                    color: '#E5E7EB',
                  }}
                >
                  <AlertTriangle size={16} style={{ color: '#F87171', flexShrink: 0, marginTop: '2px' }} />
                  <span>{risk}</span>
                </div>
              ))
            ) : (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No top risks extracted.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
