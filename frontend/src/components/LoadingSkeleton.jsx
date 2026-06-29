export default function LoadingSkeleton() {
  return (
    <div className="result-panel">
      <div className="loading-message">
        <span className="icon">🤖</span>
        Analyzing with Gemini Vision…
      </div>
      <div className="skeleton skeleton-header" />
      <div className="skeleton skeleton-card" />
      <div className="skeleton skeleton-card" />
      <div className="skeleton skeleton-summary" />
    </div>
  );
}
