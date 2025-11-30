import os
import tempfile
from typing import List


class GameVisualizer:
    """
    Generates a standalone HTML viewer for a Diplomacy game history.
    """

    def __init__(self):
        self.svg_history: List[str] = []
        self.log_history: List[str] = []

    def capture_turn(self, game_obj, logs: str = ""):
        """
        Renders the current map state to SVG and stores it.
        """
        try:
            # Fix: Remove 'incl_history' which causes the crash in standard pip version
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
                game_obj.render(output_path=tmp.name)
                tmp.close()

                with open(tmp.name, "r", encoding="utf-8") as f:
                    svg_content = f.read()

                os.unlink(tmp.name)

                self.svg_history.append(svg_content)
                self.log_history.append(logs)
        except Exception as e:
            print(f"⚠️ Visualization Warning: Could not render turn. {e}")

    def _make_path_if_not_exists(self, filepath: str):
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    def save_html(self, filepath: str):
        self._make_path_if_not_exists(os.path.dirname(filepath))
        """
        Builds the HTML file with embedded SVGs and navigation controls.
        """
        if not self.svg_history:
            print("No visual history to save.")
            return

        # Simple JS to cycle through the SVG array
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Diplomacy Simulation Replay</title>
    <style>
        body {{ font-family: sans-serif; background: #f0f0f0; text-align: center; }}
        #map-container {{ 
            width: 80%; 
            margin: 20px auto; 
            border: 1px solid #ccc; 
            background: white; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        #controls {{ margin: 20px; }}
        button {{ padding: 10px 20px; font-size: 16px; cursor: pointer; }}
        #log-display {{ 
            width: 80%; margin: 10px auto; 
            padding: 10px; background: #333; color: #0f0; 
            font-family: monospace; text-align: left; height: 100px; overflow-y: scroll; 
        }}
        #turn-info {{ font-weight: bold; font-size: 1.2em; margin: 10px; }}
    </style>
</head>
<body>
    <h1>Simulation Replay</h1>
    <div id="turn-info">Turn 0</div>
    <div id="controls">
        <button onclick="prevTurn()">❮ Prev</button>
        <span id="counter">0 / 0</span>
        <button onclick="nextTurn()">Next ❯</button>
    </div>
    
    <div id="map-container">
        </div>

    <div id="log-display">Logs appear here...</div>

    <script>
        // Embed Python data directly into JS variables
        const svgs = {self._json_list(self.svg_history)};
        const logs = {self._json_list(self.log_history)};
        let currentIndex = 0;

        function updateDisplay() {{
            document.getElementById('map-container').innerHTML = svgs[currentIndex];
            document.getElementById('log-display').innerText = logs[currentIndex] || "No logs";
            document.getElementById('counter').innerText = (currentIndex + 1) + " / " + svgs.length;
            
            // Try to extract phase from SVG title or just use index
            document.getElementById('turn-info').innerText = "Step " + (currentIndex + 1);
        }}

        function nextTurn() {{
            if (currentIndex < svgs.length - 1) {{
                currentIndex++;
                updateDisplay();
            }}
        }}

        function prevTurn() {{
            if (currentIndex > 0) {{
                currentIndex--;
                updateDisplay();
            }}
        }}

        // Init
        document.getElementById('counter').innerText = "1 / " + svgs.length;
        updateDisplay();
        
        // Key controls
        document.addEventListener('keydown', function(event) {{
            if (event.key === "ArrowRight") nextTurn();
            if (event.key === "ArrowLeft") prevTurn();
        }});
    </script>
</body>
</html>
        """

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ Visualization saved to: {filepath}")

    def _json_list(self, data: List[str]) -> str:
        """Helper to format python list as JS array literal."""
        import json

        return json.dumps(data)
