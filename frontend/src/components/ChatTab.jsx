import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Bot,
  User,
  ShieldCheck,
  AlertCircle,
  Sparkles,
  BookOpen,
  Loader2,
  CheckCircle2,
} from 'lucide-react';

// Helper component to cleanly render formatted markdown without exposing raw asterisks (**)
function FormattedMessage({ content }) {
  if (!content) return null;

  // Split content by paragraphs or double newlines
  const blocks = content.split(/\n\s*\n/);

  const renderInline = (text) => {
    if (!text) return null;

    // Tokenize for bold (**text**), citation ([Section, Page X]), and italic (*text*)
    const regex = /(\*\*[^*]+\*\*|\[[^\]]*?Page\s*\d+[^\]]*?\]|\*[^*]+\*)/g;
    const parts = text.split(regex);

    return parts.map((part, idx) => {
      if (!part) return null;

      // Bold: **text**
      if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
        return (
          <strong key={idx} className="bold-text">
            {part.slice(2, -2)}
          </strong>
        );
      }

      // Citation: [Section, Page X]
      if (part.startsWith('[') && part.endsWith(']') && /Page\s*\d+/i.test(part)) {
        return (
          <span key={idx} className="citation-pill" title="Prospectus Citation">
            <BookOpen size={11} style={{ display: 'inline', marginRight: '3px', verticalAlign: '-1px' }} />
            {part.slice(1, -1)}
          </span>
        );
      }

      // Italic: *text*
      if (part.startsWith('*') && part.endsWith('*') && part.length >= 2) {
        return (
          <em key={idx} className="italic-text">
            {part.slice(1, -1)}
          </em>
        );
      }

      return part;
    });
  };

  return (
    <div className="formatted-message">
      {blocks.map((block, bIdx) => {
        const trimmed = block.trim();
        if (!trimmed) return null;

        // Standalone Section Header: **Heading Text** or ### Heading Text
        const headingMatch = trimmed.match(/^(\*{2}|#{2,4}\s*)([^*#]+?)(\*{2})?$/);
        if (headingMatch && !trimmed.includes('\n')) {
          const headingText = headingMatch[2].trim();
          return (
            <div key={bIdx} className="message-section-title">
              <span className="title-accent" />
              <span>{headingText}</span>
            </div>
          );
        }

        // Bullet point list
        const lines = trimmed.split('\n');
        const isBulletList = lines.every((l) => /^\s*[-*•]\s+/.test(l));
        const isNumberedList = lines.every((l) => /^\s*\d+\.\s+/.test(l));

        if (isBulletList) {
          return (
            <ul key={bIdx} className="message-list">
              {lines.map((line, lIdx) => {
                const cleaned = line.replace(/^\s*[-*•]\s+/, '');
                return <li key={lIdx}>{renderInline(cleaned)}</li>;
              })}
            </ul>
          );
        }

        if (isNumberedList) {
          return (
            <ol key={bIdx} className="message-numbered-list">
              {lines.map((line, lIdx) => {
                const cleaned = line.replace(/^\s*\d+\.\s+/, '');
                return <li key={lIdx}>{renderInline(cleaned)}</li>;
              })}
            </ol>
          );
        }

        // Regulatory / Caution Alert Box
        const isNotice =
          trimmed.startsWith('⚠️') ||
          trimmed.startsWith('⚠') ||
          trimmed.toLowerCase().includes('regulatory notice:');

        return (
          <div key={bIdx} className={`message-para ${isNotice ? 'regulatory-banner' : ''}`}>
            {lines.map((line, lIdx) => (
              <React.Fragment key={lIdx}>
                {renderInline(line)}
                {lIdx < lines.length - 1 && <br />}
              </React.Fragment>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export default function ChatTab({ ipoId, companyName }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `Hello! I am your AI Prospectus Analyst for **${companyName || 'this IPO'}**. I can answer questions grounded strictly in the official DRHP/RHP filing with exact section and page citations. Ask about Use of Proceeds, Financial Multiples, Listing Risks, or Promoter Litigations!`,
      citations: [],
      numericReport: null,
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const quickPrompts = [
    'What are the objects of the issue and deployment timeline?',
    'Is potential listing gain or loss possible based on the filing?',
    'Which promoters are connected to group companies with active litigations?',
    'What are the top 3 company-specific risk factors?',
    'Show the pre-offer vs post-offer promoter shareholding.',
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (questionText) => {
    const q = (questionText || input).trim();
    if (!q || loading) return;

    setInput('');
    const userMsg = { role: 'user', content: q };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch('http://localhost:8000/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ipo_id: ipoId, question: q }),
      });

      if (!res.ok) {
        throw new Error('Query request failed.');
      }

      const data = await res.json();
      const aiMsg = {
        role: 'assistant',
        content: data.answer,
        citations: data.citations || [],
        allNumbersVerified: data.all_numbers_verified,
        numericPrecision: data.numeric_precision,
        unverifiedCount: data.unverified_numbers_count,
        isRefusal: data.is_refusal,
        isListingAnalysis: data.is_listing_analysis,
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: '⚠️ Failed to connect to the backend server. Please verify that the API is running at localhost:8000.',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Message List */}
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message-card ${msg.role}`}>
            <div className={`message-avatar ${msg.role === 'assistant' ? 'ai' : 'user'}`}>
              {msg.role === 'assistant' ? <Bot size={18} /> : <User size={18} />}
            </div>
            <div className="message-body">
              {/* Formatted Markdown Message with zero asterisks (**) */}
              <FormattedMessage content={msg.content} />

              {/* Citations & Verification Badges */}
              {msg.role === 'assistant' && (msg.citations?.length > 0 || msg.numericPrecision !== undefined) && (
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    alignItems: 'center',
                    gap: '0.45rem',
                    marginTop: '0.65rem',
                    paddingTop: '0.55rem',
                    borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                  }}
                >
                  {/* Numeric Verification Badge */}
                  {msg.allNumbersVerified !== undefined && (
                    <span
                      className={`badge ${msg.allNumbersVerified ? 'badge-emerald' : 'badge-amber'}`}
                      title={
                        msg.allNumbersVerified
                          ? 'All numbers matched source text verbatim'
                          : `${msg.unverifiedCount} numeric figures not verified verbatim in excerpts`
                      }
                    >
                      <ShieldCheck size={12} />
                      {msg.allNumbersVerified
                        ? '100% Numbers Verified'
                        : `Caution: ${msg.unverifiedCount} Unverified Fig.`}
                    </span>
                  )}

                  {/* Citation Tags */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
                      {msg.citations.map((c, idx) => (
                        <span key={idx} className="citation-pill">
                          <BookOpen size={11} style={{ display: 'inline', marginRight: '3px' }} />
                          {c.section}: Page {c.page}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Dynamic Multi-Step Loader */}
        {loading && (
          <div className="message-card assistant loading-card">
            <div className="message-avatar ai pulsing">
              <Bot size={18} />
            </div>
            <div className="message-body loading-bubble">
              <div className="loading-headline">
                <Loader2 size={16} className="spin text-blue" />
                <span className="loading-status-text">
                  Retrieving & verifying prospectus disclosures...
                </span>
              </div>
              <div className="loading-steps-row">
                <span className="step-tag active">
                  <span className="pulse-dot" /> Hybrid Vector & Lexical Search
                </span>
                <span className="step-tag">
                  BGE Cross-Encoder Reranking
                </span>
                <span className="step-tag">
                  Numeric Token Verification
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div className="chat-input-bar">
        {/* Shimmer loading bar when query is processing */}
        {loading && <div className="chat-loading-shimmer" />}

        {/* Quick Prompts - Clicking populates editable text box */}
        <div className="quick-prompts">
          {quickPrompts.map((qp, idx) => (
            <button
              key={idx}
              type="button"
              className="quick-chip"
              onClick={() => {
                setInput(qp);
                if (inputRef.current) {
                  inputRef.current.focus();
                }
              }}
              title="Click to copy into input box to edit"
            >
              <Sparkles size={11} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />
              {qp}
            </button>
          ))}
        </div>

        {/* Input Field Form */}
        <form
          className="input-row"
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
        >
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            placeholder={
              loading
                ? 'Synthesizing response from official DRHP disclosures...'
                : `Ask about ${companyName || 'this IPO'} (e.g. use of proceeds, valuation, risks)...`
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="send-btn" disabled={!input.trim() || loading}>
            {loading ? (
              <>
                <Loader2 size={15} className="spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Send size={15} />
                <span>Send</span>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
