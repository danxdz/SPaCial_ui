#!/bin/bash
# Quick 2-step deployment script for Render

echo "🚀 Quick Render Deployment"
echo "=========================="

# Step 1: Setup GitHub (if needed)
echo ""
echo "Step 1: GitHub Setup"
echo "-------------------"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
fi

# Check if we have a remote
if ! git remote -v | grep -q "origin"; then
    echo ""
    echo "🔗 No GitHub remote found!"
    echo ""
    echo "📋 Quick setup:"
    echo "1. Go to https://github.com/new"
    echo "2. Create repository: spacial-ocr-client"
    echo "3. Don't initialize with README"
    echo "4. Copy the repository URL"
    echo ""
    read -p "Enter your GitHub repository URL: " repo_url
    
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        git branch -M main
        echo "✅ Remote added!"
    else
        echo "❌ No URL provided. Exiting."
        exit 1
    fi
fi

# Add, commit, and push
echo ""
echo "📁 Adding files..."
git add .

echo "💾 Committing..."
git commit -m "Deploy to Render" || echo "No changes to commit"

echo "⬆️  Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Step 1 Complete!"
echo ""

# Step 2: Deploy to Render
echo "Step 2: Deploy to Render"
echo "----------------------"
echo ""
echo "🌐 Now deploy to Render:"
echo "1. Go to https://render.com"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repository"
echo "4. Select: spacial-ocr-client"
echo "5. Render will auto-detect settings from render.yaml"
echo "6. Click 'Deploy Web Service'"
echo ""
echo "🎉 Your app will be live in ~2 minutes!"