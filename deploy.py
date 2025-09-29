#!/usr/bin/env python3
"""
Quick deployment script for Render
This script creates a temporary git repository and pushes to GitHub for Render deployment
"""

import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def deploy_to_render():
    """Deploy the current workspace to Render via GitHub"""
    
    print("🚀 Starting Render deployment...")
    
    # Check if we're in a git repository
    is_git_repo, _, _ = run_command("git rev-parse --is-inside-work-tree")
    
    if not is_git_repo:
        print("📦 Initializing git repository...")
        success, stdout, stderr = run_command("git init")
        if not success:
            print(f"❌ Failed to initialize git: {stderr}")
            return False
    
    # Add all files
    print("📁 Adding files to git...")
    success, stdout, stderr = run_command("git add .")
    if not success:
        print(f"❌ Failed to add files: {stderr}")
        return False
    
    # Commit
    print("💾 Committing files...")
    success, stdout, stderr = run_command('git commit -m "Deploy to Render"')
    if not success:
        print(f"❌ Failed to commit: {stderr}")
        return False
    
    # Check if we have a remote
    has_remote, _, _ = run_command("git remote -v")
    
    if not has_remote:
        print("\n🔗 You need to connect this to a GitHub repository first!")
        print("\n📋 Quick setup:")
        print("1. Go to https://github.com/new")
        print("2. Create a new repository (e.g., 'spacial-ocr-client')")
        print("3. Don't initialize with README")
        print("4. Copy the repository URL")
        print("5. Run: git remote add origin <YOUR_REPO_URL>")
        print("6. Run: git branch -M main")
        print("7. Run: git push -u origin main")
        print("\nThen run this script again!")
        return False
    
    # Push to GitHub
    print("⬆️  Pushing to GitHub...")
    success, stdout, stderr = run_command("git push origin main")
    if not success:
        print(f"❌ Failed to push: {stderr}")
        return False
    
    print("\n✅ Successfully pushed to GitHub!")
    print("\n🌐 Now deploy to Render:")
    print("1. Go to https://render.com")
    print("2. Click 'New +' → 'Web Service'")
    print("3. Connect your GitHub repository")
    print("4. Select your repository")
    print("5. Render will auto-detect the settings from render.yaml")
    print("6. Click 'Deploy Web Service'")
    
    return True

if __name__ == "__main__":
    deploy_to_render()