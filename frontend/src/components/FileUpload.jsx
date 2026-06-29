import { useState, useRef } from 'react';

export default function FileUpload({ accept, label, hint, onFileSelected, file }) {
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) onFileSelected(dropped);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleChange = (e) => {
    const picked = e.target.files[0];
    if (picked) onFileSelected(picked);
  };

  const handleRemove = (e) => {
    e.stopPropagation();
    onFileSelected(null);
    if (fileInput.current) fileInput.current.value = '';
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div>
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={() => setDragOver(false)}
        onClick={() => fileInput.current?.click()}
      >
        <input
          ref={fileInput}
          type="file"
          accept={accept}
          onChange={handleChange}
          style={{ display: 'none' }}
        />
        <span className="upload-icon">
          {accept?.includes('pdf') ? '📄' : '🖼️'}
        </span>
        <div className="upload-text">
          <strong>Click to upload</strong> or drag and drop
        </div>
        <div className="upload-hint">{hint || label}</div>
      </div>

      {file && (
        <div className="file-info fade-in">
          <span className="file-info-icon">
            {file.name.endsWith('.pdf') ? '📄' : '🖼️'}
          </span>
          <div className="file-info-details">
            <div className="file-info-name">{file.name}</div>
            <div className="file-info-meta">{formatSize(file.size)}</div>
          </div>
          <button className="file-remove-btn" onClick={handleRemove} title="Remove file">
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
