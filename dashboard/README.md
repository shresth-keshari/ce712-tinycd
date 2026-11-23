# Supplementary static dashboard

The primary app is Streamlit: `streamlit run app/app.py`. This optional HTML/JavaScript dashboard displays the same frozen-checkpoint benchmark from a generated JSON export. Refresh it with `python scripts/export_dashboard.py` and serve it locally using `python -m http.server 8000 --directory dashboard/dist`; open http://localhost:8000. Its HTML includes no inference backend. It does not contain Powai predictions or any locally trained-model leaderboard.
