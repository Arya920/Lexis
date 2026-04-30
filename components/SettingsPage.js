function SettingsPage() {
  return (
    <div className="page-wrap">
      <h1>Settings</h1>

      <div className="card">
        <h3>Appearance</h3>
        <button onClick={() => document.body.classList.toggle("light")}>
          Toggle Theme
        </button>
      </div>

      <div className="card">
        <h3>Model</h3>
        <select>
          <option>Llama 3.1 8B</option>
          <option>Llama 3.3 70B</option>
        </select>
      </div>
    </div>
  );
}