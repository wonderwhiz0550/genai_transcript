Resume Python Project Guide
=========================
This guide helps you resume your Python project in /Users/shubhadeepdas/Documents/data_science/projects/genai_transcript with JupyterLab and GitHub (https://github.com/svdp2304/genai_transcript).

1. Navigate to Project Folder
   - Command: cd /Users/shubhadeepdas/Documents/data_science/projects/genai_transcript
   - Purpose: Sets the working directory to your project.

2. Activate Virtual Environment
   - Command: source venv/bin/activate
   - Purpose: Activates the project's isolated Python environment.

3. Update Dependencies
   - Command: pip install --upgrade pip jupyterlab jupyterlab-git pandas numpy
   - Purpose: Ensures JupyterLab, Git extension, and key libraries are up-to-date.

4. Start JupyterLab
   - Command: jupyter lab
   - Purpose: Opens JupyterLab in your browser for notebook editing.

5. Create/Edit Notebooks
   - Action: In JupyterLab, go to File > New > Notebook to create a new .ipynb file, or open an existing one (e.g., genai_transcript.ipynb). Save changes.
   - Purpose: Work on your Python code and analysis in notebooks.

6. Sync with GitHub
   - Commands (in Terminal):
     git pull origin main --rebase  # Fetches latest remote changes
     git add <file>                 # Stages changed files (e.g., genai_transcript.ipynb)
     git commit -m "Update notebook" # Commits changes
     git push -u origin main        # Pushes to GitHub
   - Alternative: Use JupyterLab's Git panel (left sidebar) to stage, commit, and push.
   - Purpose: Keeps your local and GitHub repositories in sync.

7. Handle Git Conflicts
   - Action: If git pull reports conflicts, edit conflicted files in JupyterLab or a text editor, resolve marked sections, then:
     git add <file>
     git rebase --continue
     git push -u origin main
   - Purpose: Resolves any overlapping changes with the remote repo.

8. Verify on GitHub
   - Action: Visit https://github.com/svdp2304/genai_transcript
   - Purpose: Confirms your changes (e.g., notebooks) are uploaded.

Notes:
- If authentication fails, regenerate a Personal Access Token at https://github.com/settings/tokens (select 'repo' scope) and use it when prompted.
- If the GitHub repo uses 'master' instead of 'main', replace 'main' with 'master' in Git commands.
- Deactivate the virtual environment when done: deactivate
- For errors, check git status, git branch, or contact support with error messages.