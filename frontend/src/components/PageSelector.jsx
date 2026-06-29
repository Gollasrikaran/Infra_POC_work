export default function PageSelector({ totalPages, selectedPage, onPageChange, pagesList = [] }) {
  return (
    <div className="page-selector">
      <label htmlFor="page-select">Select Page</label>
      <select
        id="page-select"
        value={selectedPage}
        onChange={(e) => onPageChange(Number(e.target.value))}
      >
        {Array.from({ length: totalPages }, (_, i) => i + 1).map((num) => (
          <option key={num} value={num}>
            Page {num}
          </option>
        ))}
      </select>
    </div>
  );
}
