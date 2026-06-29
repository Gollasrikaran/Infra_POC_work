import { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import LoginPage from './components/LoginPage';
import FileUpload from './components/FileUpload';
import MetricCard from './components/MetricCard';
import PageSelector from './components/PageSelector';
import AnalysisResult from './components/AnalysisResult';
import LoadingSkeleton from './components/LoadingSkeleton';

const API_BASE = '/api';

// --- Home Page ---

function HomePage({ onNavigate }) {
  return (
    <div className="fade-in">
      <div className="breadcrumb">
        <a href="#" onClick={(e) => { e.preventDefault(); onNavigate('home'); }}>Home</a>
        <span className="sep">›</span>
        <span className="breadcrumb-current">Dashboard</span>
      </div>
      <h1 className="page-title">Cross-Section Analyzer</h1>

      <div className="home-banner fade-in-up">
        <div className="home-banner-content">
          <div className="home-banner-badge">🛣️ Highway Engineering AI</div>
          <h2 className="home-banner-heading">
            Analyze cross-sections<br />with Gemini Vision AI
          </h2>
          <p className="home-banner-desc">
            Upload highway engineering PDFs or images. Our AI identifies cut/fill regions,
            calculates trapezoidal areas, and extracts station data — instantly.
          </p>
          <button className="home-banner-btn" onClick={() => onNavigate('analyze')}>
            Get Started →
          </button>
        </div>
        <div className="home-banner-visual">
          <div className="home-visual-graphic">
            <div className="visual-line visual-line-1" />
            <div className="visual-line visual-line-2" />
            <div className="visual-fill-area" />
            <div className="visual-cut-area" />
            <div className="visual-label cut-label">CUT</div>
            <div className="visual-label fill-label">FILL</div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="home-stats fade-in-up" style={{ animationDelay: '0.1s' }}>
        <div className="home-stat-card">
          <div className="home-stat-icon" style={{ background: '#e3f2fd' }}>📄</div>
          <div>
            <div className="home-stat-label">Supported Formats</div>
            <div className="home-stat-value">PDF, PNG, JPG</div>
          </div>
        </div>
        <div className="home-stat-card">
          <div className="home-stat-icon" style={{ background: '#e8f5e9' }}>🤖</div>
          <div>
            <div className="home-stat-label">AI Engine</div>
            <div className="home-stat-value">Gemini 2.5 Flash</div>
          </div>
        </div>
        <div className="home-stat-card">
          <div className="home-stat-icon" style={{ background: '#fff3e0' }}>📐</div>
          <div>
            <div className="home-stat-label">Area Method</div>
            <div className="home-stat-value">Shoelace Formula</div>
          </div>
        </div>
        <div className="home-stat-card">
          <div className="home-stat-icon" style={{ background: '#f3e5f5' }}>⚡</div>
          <div>
            <div className="home-stat-label">Analysis Speed</div>
            <div className="home-stat-value">~10 seconds</div>
          </div>
        </div>
      </div>

      {/* Feature cards */}
      <div className="home-features fade-in-up" style={{ animationDelay: '0.15s' }}>
        <div className="home-feature-card" onClick={() => onNavigate('analyze')}>
          <div className="home-feature-icon-wrap green">
            <span>📄</span>
          </div>
          <div className="home-feature-title">PDF Analysis</div>
          <div className="home-feature-desc">
            Upload multi-page engineering PDFs. Select any page, render it, and analyze
            cut/fill regions with AI vision.
          </div>
          <div className="home-feature-action">Start analyzing →</div>
        </div>

        <div className="home-feature-card" onClick={() => onNavigate('analyze')}>
          <div className="home-feature-icon-wrap blue">
            <span>🖼️</span>
          </div>
          <div className="home-feature-title">Image Analysis</div>
          <div className="home-feature-desc">
            Drop a PNG or JPG cross-section image directly. Get instant CUT/FILL
            identification and area calculations.
          </div>
          <div className="home-feature-action">Upload image →</div>
        </div>

        <div className="home-feature-card">
          <div className="home-feature-icon-wrap purple">
            <span>📊</span>
          </div>
          <div className="home-feature-title">Detailed Reports</div>
          <div className="home-feature-desc">
            View station info, catch points, slope ratios, sample elevations, and
            net cut/fill summaries for every intersection.
          </div>
          <div className="home-feature-action">View sample →</div>
        </div>
      </div>

      {/* How it works */}
      <div className="home-how fade-in-up" style={{ animationDelay: '0.2s' }}>
        <div className="card-header">
          <div className="card-title">How It Works</div>
        </div>
        <div className="home-steps">
          <div className="home-step">
            <div className="home-step-num">1</div>
            <div className="home-step-title">Upload</div>
            <div className="home-step-desc">Drop a PDF or image of a highway cross-section drawing</div>
          </div>
          <div className="home-step-arrow">→</div>
          <div className="home-step">
            <div className="home-step-num">2</div>
            <div className="home-step-title">Analyze</div>
            <div className="home-step-desc">Gemini Vision reads lines, slopes, and elevations from the drawing</div>
          </div>
          <div className="home-step-arrow">→</div>
          <div className="home-step">
            <div className="home-step-num">3</div>
            <div className="home-step-title">Results</div>
            <div className="home-step-desc">View CUT/FILL regions, areas, catch points, and net calculations</div>
          </div>
        </div>
      </div>
    </div>
  );
}


// --- Main App ---

function App() {
  // auth state
  const [user, setUser] = useState(() => {
    try {
      const stored = localStorage.getItem('cs_user');
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const handleLogin = (userData) => setUser(userData);

  const handleLogout = () => {
    localStorage.removeItem('cs_user');
    setUser(null);
  };

  // nav + UI state
  const [currentPage, setCurrentPage] = useState('home');
  const [activeTab, setActiveTab] = useState('pdf');
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const profileRef = useRef(null);

  // close profile dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (profileRef.current && !profileRef.current.contains(e.target)) {
        setShowProfileMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // lightbox for zoomed images
  const [zoomedImage, setZoomedImage] = useState(null);

  // PDF analysis state
  const [pdfFile, setPdfFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [totalPages, setTotalPages] = useState(0);
  const [pagesList, setPagesList] = useState([]);
  const [selectedPage, setSelectedPage] = useState(1);
  const [pageImageUrl, setPageImageUrl] = useState(null);
  const [pdfResult, setPdfResult] = useState(null);
  const [pdfError, setPdfError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzingPdf, setIsAnalyzingPdf] = useState(false);

  // image analysis state
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [coloredImagePreview, setColoredImagePreview] = useState(null);
  const [isColorizing, setIsColorizing] = useState(false);
  const [imageResult, setImageResult] = useState(null);
  const [imageError, setImageError] = useState(null);
  const [isAnalyzingImage, setIsAnalyzingImage] = useState(false);

  // Crop state
  const [isCropMode, setIsCropMode] = useState(false);
  const [cropStart, setCropStart] = useState(null);
  const [cropEnd, setCropEnd] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [croppedBlob, setCroppedBlob] = useState(null);
  const [croppedPreview, setCroppedPreview] = useState(null);
  const lightboxImgRef = useRef(null);

  // close lightbox on Escape key
  const closeLightbox = useCallback(() => setZoomedImage(null), []);
  useEffect(() => {
    if (!zoomedImage) return;
    const handler = (e) => { if (e.key === 'Escape') closeLightbox(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [zoomedImage, closeLightbox]);

  // --- Crop helpers ---
  const getMousePosOnImage = (e) => {
    const img = lightboxImgRef.current;
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(e.clientX - rect.left, rect.width)),
      y: Math.max(0, Math.min(e.clientY - rect.top, rect.height)),
    };
  };

  const handleCropMouseDown = (e) => {
    if (!isCropMode) return;
    e.preventDefault();
    e.stopPropagation();
    const pos = getMousePosOnImage(e);
    if (pos) {
      setCropStart(pos);
      setCropEnd(pos);
      setIsDragging(true);
    }
  };

  const handleCropMouseMove = (e) => {
    if (!isDragging || !isCropMode) return;
    e.preventDefault();
    const pos = getMousePosOnImage(e);
    if (pos) setCropEnd(pos);
  };

  const handleCropMouseUp = (e) => {
    if (!isDragging) return;
    e.preventDefault();
    setIsDragging(false);
  };

  const applyCrop = () => {
    const img = lightboxImgRef.current;
    if (!img || !cropStart || !cropEnd) return;

    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;

    const sx = Math.min(cropStart.x, cropEnd.x) * scaleX;
    const sy = Math.min(cropStart.y, cropEnd.y) * scaleY;
    const sw = Math.abs(cropEnd.x - cropStart.x) * scaleX;
    const sh = Math.abs(cropEnd.y - cropStart.y) * scaleY;

    if (sw < 10 || sh < 10) return;

    const canvas = document.createElement('canvas');
    canvas.width = sw;
    canvas.height = sh;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

    canvas.toBlob((blob) => {
      if (blob) {
        setCroppedBlob(blob);
        setCroppedPreview(canvas.toDataURL('image/png'));
        setIsCropMode(false);
        setCropStart(null);
        setCropEnd(null);
        closeLightbox();
      }
    }, 'image/png');
  };

  const cancelCrop = () => {
    setIsCropMode(false);
    setCropStart(null);
    setCropEnd(null);
    setIsDragging(false);
  };

  const clearCrop = () => {
    setCroppedBlob(null);
    setCroppedPreview(null);
  };

  // --- PDF upload handler ---

  const handlePdfSelected = async (file) => {
    setPdfResult(null);
    setPdfError(null);
    setPageImageUrl(null);
    setFileId(null);
    setTotalPages(0);
    setSelectedPage(1);
    setCroppedBlob(null);
    setCroppedPreview(null);

    if (!file) {
      setPdfFile(null);
      return;
    }

    setPdfFile(file);
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API_BASE}/upload-pdf`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      setFileId(data.file_id);
      setTotalPages(data.total_pages);
      setPagesList(data.pages || []);
      setSelectedPage(1);
    } catch (err) {
      setPdfError(err.message);
      setPdfFile(null);
    } finally {
      setIsUploading(false);
    }
  };

  // update preview when page changes
  useEffect(() => {
    if (!fileId || !selectedPage) {
      setPageImageUrl(null);
      return;
    }
    setPageImageUrl(`${API_BASE}/pdf/${fileId}/page/${selectedPage}/colored`);
    setPdfResult(null);
    setPdfError(null);
    setCroppedBlob(null);
    setCroppedPreview(null);
  }, [fileId, selectedPage]);

  // --- PDF analysis handler ---

  const handleAnalyzePdf = async () => {
    if (!fileId || !selectedPage) return;

    setIsAnalyzingPdf(true);
    setPdfResult(null);
    setPdfError(null);

    try {
      const requests = [];

      // If we have a cropped image, send it as an image analysis instead
      if (croppedBlob) {
        const formData = new FormData();
        formData.append('file', croppedBlob, 'cropped_page.png');
        requests.push(
          fetch(`${API_BASE}/analyze/image`, {
            method: 'POST',
            body: formData,
          }).then(async (res) => {
            if (!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || 'Analysis failed');
            }
            return res.json();
          })
        );
      } else {
        requests.push(
          fetch(`${API_BASE}/analyze/pdf-page`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_id: fileId, page_num: selectedPage }),
          }).then(async (res) => {
            if (!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || 'Analysis failed');
            }
            return res.json();
          })
        );
      }

      const [currentRes] = await Promise.all(requests);

      if (currentRes.success && currentRes.data) {
        const combinedData = { ...currentRes.data };
        
        // Retrieve previous page's station directly from local cache (pagesList)
        if (selectedPage > 1 && pagesList[selectedPage - 2]) {
          combinedData.previous_station = pagesList[selectedPage - 2].station;
        } else {
          combinedData.previous_station = null;
        }
        
        // Fallback for current station if Gemini output is missing or empty
        if (!combinedData.station && pagesList[selectedPage - 1]) {
          combinedData.station = pagesList[selectedPage - 1].station;
        }

        setPdfResult(combinedData);
      } else {
        setPdfError(currentRes.error || 'Could not parse Gemini response');
        if (currentRes.raw) setPdfResult({ _raw: currentRes.raw });
      }
    } catch (err) {
      setPdfError(err.message);
    } finally {
      setIsAnalyzingPdf(false);
    }
  };

  // --- Image upload + analysis ---

  const handleImageSelected = async (file) => {
    setImageResult(null);
    setImageError(null);
    setImagePreview(null);
    setColoredImagePreview(null);
    setCroppedBlob(null);
    setCroppedPreview(null);

    if (!file) {
      setImageFile(null);
      return;
    }

    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));

    // Colorize the image via OpenCV backend
    setIsColorizing(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/colorize-image`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const blob = await res.blob();
        setColoredImagePreview(URL.createObjectURL(blob));
      }
    } catch (err) {
      console.error('Image colorization failed:', err);
    } finally {
      setIsColorizing(false);
    }
  };

  const handleAnalyzeImage = async () => {
    if (!imageFile) return;

    setIsAnalyzingImage(true);
    setImageResult(null);
    setImageError(null);

    try {
      const formData = new FormData();
      // If we have a cropped blob, send that instead of the full image
      if (croppedBlob) {
        formData.append('file', croppedBlob, 'cropped_image.png');
      } else {
        formData.append('file', imageFile);
      }

      const res = await fetch(`${API_BASE}/analyze/image`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis failed');
      }

      const data = await res.json();
      if (data.success && data.data) {
        setImageResult(data.data);
      } else {
        setImageError(data.error || 'Could not parse Gemini response');
        if (data.raw) setImageResult({ _raw: data.raw });
      }
    } catch (err) {
      setImageError(err.message);
    } finally {
      setIsAnalyzingImage(false);
    }
  };

  // --- Render ---

  if (!user) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">🛣️</div>

        <button
          className={`sidebar-btn ${currentPage === 'home' ? 'active' : ''}`}
          onClick={() => setCurrentPage('home')}
          title="Home"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
            <polyline points="9 22 9 12 15 12 15 22" />
          </svg>
          <span className="sidebar-btn-label">Home</span>
        </button>

        <button
          className={`sidebar-btn ${currentPage === 'analyze' ? 'active' : ''}`}
          onClick={() => setCurrentPage('analyze')}
          title="Analyze"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="12" y1="18" x2="12" y2="12" />
            <line x1="9" y1="15" x2="15" y2="15" />
          </svg>
          <span className="sidebar-btn-label">Upload</span>
        </button>

        <div className="sidebar-spacer" />

        <button className="sidebar-btn" title="Help">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <span className="sidebar-btn-label">Help</span>
        </button>
      </aside>

      {/* Profile menu (top right) */}
      <div className="profile-bar">
        <div className="profile-widget" ref={profileRef}>
          <button
            className="profile-trigger"
            onClick={() => setShowProfileMenu((v) => !v)}
            title={user.name}
          >
            <svg className="profile-trigger-menu" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
            <div className="profile-avatar">
              {user.name.charAt(0).toUpperCase()}
            </div>
          </button>

          {showProfileMenu && (
            <div className="profile-dropdown fade-in">
              <div className="profile-dropdown-header">
                <div className="profile-dropdown-avatar">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <div className="profile-dropdown-info">
                  <div className="profile-dropdown-name">{user.name}</div>
                  <div className="profile-dropdown-email">{user.email}</div>
                </div>
              </div>
              <div className="profile-dropdown-divider" />
              <button className="profile-dropdown-item" onClick={() => setShowProfileMenu(false)}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                  <circle cx="12" cy="7" r="4" />
                </svg>
                Account Info
              </button>
              <button className="profile-dropdown-item" onClick={() => setShowProfileMenu(false)}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
                Settings
              </button>
              <div className="profile-dropdown-divider" />
              <button className="profile-dropdown-item profile-dropdown-logout" onClick={handleLogout}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                  <polyline points="16 17 21 12 16 7" />
                  <line x1="21" y1="12" x2="9" y2="12" />
                </svg>
                Logout
              </button>
            </div>
          )}
        </div>
      </div>

      <main className="main-content">
        {/* Home page */}
        {currentPage === 'home' && (
          <HomePage onNavigate={setCurrentPage} />
        )}

        {/* Analyze page */}
        {currentPage === 'analyze' && (
          <div className="fade-in">
            <div className="top-bar">
              <div className="top-tabs">
                <button
                  className={`top-tab ${activeTab === 'pdf' ? 'active' : ''}`}
                  onClick={() => setActiveTab('pdf')}
                >
                  <span className="top-tab-icon">📄</span> PDF Analysis
                </button>
                <button
                  className={`top-tab ${activeTab === 'image' ? 'active' : ''}`}
                  onClick={() => setActiveTab('image')}
                >
                  <span className="top-tab-icon">🖼️</span> Image Analysis
                </button>
              </div>
              <div className="top-actions">
                <button className="top-action-btn">⚡ Gemini Flash</button>
              </div>
            </div>

            <div className="breadcrumb">
              <a href="#" onClick={(e) => { e.preventDefault(); setCurrentPage('home'); }}>Home</a>
              <span className="sep">›</span>
              <span>{activeTab === 'pdf' ? 'PDF Analysis' : 'Image Analysis'}</span>
            </div>
            <h1 className="page-title">Cross-Section Analyzer</h1>

            {/* PDF tab */}
            {activeTab === 'pdf' && (
              <div className="fade-in">
                {!fileId && (
                  <div className="card-grid">
                    <div className="card full-width" style={{ maxWidth: 600, margin: '0 auto' }}>
                      <div className="card-header">
                        <div>
                          <div className="card-title">Upload PDF</div>
                          <div className="card-subtitle">Highway engineering cross-section drawings</div>
                        </div>
                      </div>
                      <FileUpload
                        accept=".pdf"
                        label="Upload PDF"
                        hint="PDF files — highway engineering drawings"
                        onFileSelected={handlePdfSelected}
                        file={pdfFile}
                      />
                      {isUploading && (
                        <div className="loading-message" style={{ marginTop: '1rem' }}>
                          <span className="icon">📄</span>
                          Uploading and processing…
                        </div>
                      )}
                      {pdfError && !fileId && (
                        <div className="error-box">
                          <div className="title">Upload Error</div>
                          <div className="message">{pdfError}</div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {fileId && totalPages > 0 && (
                  <>
                    <div className="metrics-row fade-in-up">
                      <MetricCard value={totalPages} label="Total Pages" />
                      <MetricCard value={totalPages * 5} label="Est. Sections" />
                      <MetricCard value="Gemini + CV" label="Engine" small />
                    </div>

                    <div className="content-grid">
                      <div className="card fade-in-up">
                        <div className="card-header">
                          <div>
                            <div className="card-title">Page Selection</div>
                            <div className="card-subtitle">{pdfFile?.name}</div>
                          </div>
                          <button
                            className="card-icon-btn"
                            title="Change PDF"
                            onClick={() => handlePdfSelected(null)}
                          >✕</button>
                        </div>

                        <PageSelector
                          totalPages={totalPages}
                          selectedPage={selectedPage}
                          onPageChange={setSelectedPage}
                          pagesList={pagesList}
                        />

                        {pageImageUrl && (
                          <div className="page-preview">
                            <img src={pageImageUrl} alt={`Page ${selectedPage}`} loading="lazy" />
                            <button
                              className="preview-maximize-btn"
                              title="Maximize"
                              onClick={() => setZoomedImage(pageImageUrl)}
                            >
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="15 3 21 3 21 9" />
                                <polyline points="9 21 3 21 3 15" />
                                <line x1="21" y1="3" x2="14" y2="10" />
                                <line x1="3" y1="21" x2="10" y2="14" />
                              </svg>
                            </button>
                          </div>
                        )}

                        {croppedPreview && activeTab === 'pdf' && (
                          <div className="crop-indicator">
                            <div className="crop-indicator-header">
                              <span>✂️ Cropped Region</span>
                              <button className="crop-clear-btn" onClick={clearCrop} title="Remove crop">✕</button>
                            </div>
                            <div className="crop-indicator-preview">
                              <img src={croppedPreview} alt="Cropped region" />
                            </div>
                            <div className="crop-indicator-info">Only this cropped region will be analyzed</div>
                          </div>
                        )}

                        <button
                          className="btn-analyze"
                          onClick={handleAnalyzePdf}
                          disabled={isAnalyzingPdf || !fileId}
                        >
                          {isAnalyzingPdf ? (
                            <><span className="btn-spinner" /> Analyzing…</>
                          ) : (
                            <>🔍 Analyze This Page</>
                          )}
                        </button>
                      </div>

                      <div>
                        <div className="panel-label">Analysis Result</div>
                        {isAnalyzingPdf && <LoadingSkeleton />}
                        {pdfError && !isAnalyzingPdf && (
                          <div className="error-box">
                            <div className="title">Analysis Error</div>
                            <div className="message">{pdfError}</div>
                          </div>
                        )}
                        {pdfResult && !pdfResult._raw && !isAnalyzingPdf && (
                          <AnalysisResult data={pdfResult} />
                        )}
                        {pdfResult?._raw && !isAnalyzingPdf && (
                          <div>
                            <div className="error-box">
                              <div className="title">⚠ Could not parse JSON</div>
                              <div className="message">Raw Gemini response below</div>
                            </div>
                            <pre className="raw-json" style={{ marginTop: '12px' }}>{pdfResult._raw}</pre>
                          </div>
                        )}
                        {!pdfResult && !isAnalyzingPdf && !pdfError && (
                          <div className="empty-state">
                            <span className="empty-state-icon">📊</span>
                            <p>Select a page and click<br /><strong>Analyze This Page</strong></p>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Image tab */}
            {activeTab === 'image' && (
              <div className="fade-in">
                {!imageFile && (
                  <div className="card-grid">
                    <div className="card full-width" style={{ maxWidth: 600, margin: '0 auto' }}>
                      <div className="card-header">
                        <div>
                          <div className="card-title">Upload Image</div>
                          <div className="card-subtitle">PNG or JPG cross-section image</div>
                        </div>
                      </div>
                      <FileUpload
                        accept=".png,.jpg,.jpeg"
                        label="Upload Image"
                        hint="PNG, JPG — cross-section images"
                        onFileSelected={handleImageSelected}
                        file={imageFile}
                      />
                    </div>
                  </div>
                )}

                {imageFile && (
                  <div className="content-grid">
                    <div className="card fade-in-up">
                      <div className="card-header">
                        <div>
                          <div className="card-title">Uploaded Image</div>
                          <div className="card-subtitle">{imageFile.name}</div>
                        </div>
                        <button
                          className="card-icon-btn"
                          title="Remove"
                          onClick={() => handleImageSelected(null)}
                        >✕</button>
                      </div>

                      {isColorizing && (
                        <div className="loading-message" style={{ marginBottom: '0.5rem' }}>
                          <span className="icon">🎨</span>
                          Colorizing cross-sections…
                        </div>
                      )}

                      {(coloredImagePreview || imagePreview) && (
                        <div className="image-preview">
                          <img src={coloredImagePreview || imagePreview} alt={imageFile.name} />
                          <button
                            className="preview-maximize-btn"
                            title="Maximize"
                            onClick={() => setZoomedImage(coloredImagePreview || imagePreview)}
                          >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                              <polyline points="15 3 21 3 21 9" />
                              <polyline points="9 21 3 21 3 15" />
                              <line x1="21" y1="3" x2="14" y2="10" />
                              <line x1="3" y1="21" x2="10" y2="14" />
                            </svg>
                          </button>
                        </div>
                      )}

                      {croppedPreview && activeTab === 'image' && (
                        <div className="crop-indicator">
                          <div className="crop-indicator-header">
                            <span>✂️ Cropped Region</span>
                            <button className="crop-clear-btn" onClick={clearCrop} title="Remove crop">✕</button>
                          </div>
                          <div className="crop-indicator-preview">
                            <img src={croppedPreview} alt="Cropped region" />
                          </div>
                          <div className="crop-indicator-info">Only this cropped region will be analyzed</div>
                        </div>
                      )}

                      <button
                        className="btn-analyze"
                        onClick={handleAnalyzeImage}
                        disabled={isAnalyzingImage || !imageFile}
                      >
                        {isAnalyzingImage ? (
                          <><span className="btn-spinner" /> Analyzing…</>
                        ) : (
                          <>🔍 Analyze with Gemini</>
                        )}
                      </button>
                    </div>

                    <div>
                      <div className="panel-label">Gemini Vision Analysis</div>
                      {isAnalyzingImage && <LoadingSkeleton />}
                      {imageError && !isAnalyzingImage && (
                        <div className="error-box">
                          <div className="title">Analysis Error</div>
                          <div className="message">{imageError}</div>
                        </div>
                      )}
                      {imageResult && !imageResult._raw && !isAnalyzingImage && (
                        <AnalysisResult data={imageResult} />
                      )}
                      {imageResult?._raw && !isAnalyzingImage && (
                        <div>
                          <div className="error-box">
                            <div className="title">⚠ Could not parse JSON</div>
                            <div className="message">Raw Gemini response below</div>
                          </div>
                          <pre className="raw-json" style={{ marginTop: '12px' }}>{imageResult._raw}</pre>
                        </div>
                      )}
                      {!imageResult && !isAnalyzingImage && !imageError && (
                        <div className="empty-state">
                          <span className="empty-state-icon">📊</span>
                          <p>Upload an image and click<br /><strong>Analyze with Gemini</strong></p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Lightbox overlay with crop support */}
      {zoomedImage && (
        <div
          className={`lightbox-overlay ${isCropMode ? 'crop-active' : ''}`}
          onClick={isCropMode ? undefined : closeLightbox}
          onMouseMove={isCropMode ? handleCropMouseMove : undefined}
          onMouseUp={isCropMode ? handleCropMouseUp : undefined}
        >
          {/* Close button */}
          <button className="lightbox-close" onClick={() => { cancelCrop(); closeLightbox(); }} title="Close (Esc)">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>

          {/* Crop toolbar */}
          <div className="lightbox-toolbar" onClick={(e) => e.stopPropagation()}>
            {!isCropMode ? (
              <button className="lightbox-tool-btn" onClick={() => setIsCropMode(true)} title="Crop Image">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M6.13 1L6 16a2 2 0 0 0 2 2h15" />
                  <path d="M1 6.13L16 6a2 2 0 0 1 2 2v15" />
                </svg>
                <span>Crop</span>
              </button>
            ) : (
              <>
                <button
                  className="lightbox-tool-btn confirm"
                  onClick={applyCrop}
                  disabled={!cropStart || !cropEnd || Math.abs((cropEnd?.x || 0) - (cropStart?.x || 0)) < 10}
                  title="Apply Crop"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>Use Cropped</span>
                </button>
                <button className="lightbox-tool-btn cancel" onClick={cancelCrop} title="Cancel Crop">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                  <span>Cancel</span>
                </button>
              </>
            )}
          </div>

          {/* Crop mode instructions */}
          {isCropMode && !cropStart && (
            <div className="crop-instructions" onClick={(e) => e.stopPropagation()}>
              Click and drag on the image to select a crop region
            </div>
          )}

          {/* Image container with crop overlay */}
          <div
            className="lightbox-img-container"
            onClick={(e) => e.stopPropagation()}
            onMouseDown={isCropMode ? handleCropMouseDown : undefined}
            style={{ cursor: isCropMode ? 'crosshair' : 'default' }}
          >
            <img
              ref={lightboxImgRef}
              className="lightbox-img"
              src={zoomedImage}
              alt="Maximized view"
              draggable={false}
              style={{ userSelect: 'none' }}
            />
            {/* Crop selection rectangle */}
            {isCropMode && cropStart && cropEnd && (
              <>
                {/* Dark overlay outside crop */}
                <div className="crop-overlay-mask" style={{
                  clipPath: `polygon(
                    0% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%,
                    ${Math.min(cropStart.x, cropEnd.x)}px ${Math.min(cropStart.y, cropEnd.y)}px,
                    ${Math.min(cropStart.x, cropEnd.x)}px ${Math.max(cropStart.y, cropEnd.y)}px,
                    ${Math.max(cropStart.x, cropEnd.x)}px ${Math.max(cropStart.y, cropEnd.y)}px,
                    ${Math.max(cropStart.x, cropEnd.x)}px ${Math.min(cropStart.y, cropEnd.y)}px,
                    ${Math.min(cropStart.x, cropEnd.x)}px ${Math.min(cropStart.y, cropEnd.y)}px
                  )`
                }} />
                {/* Crop border */}
                <div className="crop-selection" style={{
                  left: Math.min(cropStart.x, cropEnd.x),
                  top: Math.min(cropStart.y, cropEnd.y),
                  width: Math.abs(cropEnd.x - cropStart.x),
                  height: Math.abs(cropEnd.y - cropStart.y),
                }}>
                  <div className="crop-handle crop-handle-tl" />
                  <div className="crop-handle crop-handle-tr" />
                  <div className="crop-handle crop-handle-bl" />
                  <div className="crop-handle crop-handle-br" />
                  <div className="crop-size">
                    {Math.round(Math.abs(cropEnd.x - cropStart.x))} × {Math.round(Math.abs(cropEnd.y - cropStart.y))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;