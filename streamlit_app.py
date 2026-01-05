import streamlit as st
import os
from collections import defaultdict



st.set_page_config(layout="wide")

app_dir = os.path.join(os.path.dirname(__file__), "app")

if not os.path.exists(app_dir):
    st.error("The 'app' folder does not exist!")
else:
    py_files = [f for f in os.listdir(app_dir) if f.endswith('.py') and not f.startswith('_')]

    groups = defaultdict(list)
    for file in py_files:
        if "_" in file:
            parts = file.split("_")
            if len(parts) >= 3:
                order = parts[0]
                group = parts[1]
                page = "_".join(parts[2:])
                groups[group].append((int(order), page, file))

    allowed_groups = ['Home', 'Stock', 'Portfolio']
    filtered_groups = {group: groups[group] for group in allowed_groups if group in groups}
    
    for group in filtered_groups:
        filtered_groups[group].sort(key=lambda x: x[0])

    pages = {}
    for group, files in filtered_groups.items():
        pages[group] = []
        for _, page, file in files:
            title = page.replace("_", " ").replace(".py", "").title()
            file_path = os.path.join(app_dir, file)
            pages[group].append(st.Page(file_path, title=title))

    pg = st.navigation(pages)
    pg.run()