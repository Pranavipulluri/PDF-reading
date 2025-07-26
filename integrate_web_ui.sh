#!/bin/bash

# Integrate Web UI Script - One-click integration
# Adds beautiful web interface to your PDF outline extractor

echo "🌐 PDF Outline Extractor - Web UI Integration"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "src/main.py" ] && [ ! -f "main.py" ]; then
    echo "❌ Please run this script from your pdf-outline-extractor directory"
    echo "   (The directory should contain src/main.py or main.py)"
    exit 1
fi

echo "✅ Found project directory"

# Backup existing files
echo "📁 Creating backups..."
if [ -f "requirements.txt" ]; then
    cp requirements.txt requirements.txt.backup
    echo "   Backed up requirements.txt"
fi

if [ -f "Dockerfile" ]; then
    cp Dockerfile Dockerfile.backup  
    echo "   Backed up Dockerfile"
fi

# Add web server
echo "🌐 Adding web server..."
if [ ! -f "web_server.py" ]; then
    echo "❌ web_server.py not found. Please ensure you have the web_server.py file."
    exit 1
fi

echo "✅ Web server file ready"

# Update requirements
echo "📦 Updating requirements..."
if [ -f "requirements.web.txt" ]; then
    cp requirements.web.txt requirements.txt
    echo "✅ Updated requirements with Flask support"
else
    echo "⚠️  requirements.web.txt not found, adding Flask to existing requirements"
    echo "Flask==2.3.3" >> requirements.txt
    echo "Werkzeug==2.3.7" >> requirements.txt
    echo "✅ Added Flask to requirements"
fi

# Update Dockerfile if available
if [ -f "Dockerfile.web" ]; then
    echo "🐳 Updating Dockerfile for web support..."
    cp Dockerfile.web Dockerfile
    echo "✅ Updated Dockerfile"
else
    echo "⚠️  Dockerfile.web not found, keeping existing Dockerfile"
fi

# Check Docker
echo "🐳 Checking Docker..."
if docker --version >/dev/null 2>&1; then
    echo "✅ Docker is available"
    
    echo "🔨 Building web-enabled image..."
    if docker build --platform linux/amd64 -t pdf-extractor-web:latest .; then
        echo "✅ Docker build successful!"
        
        echo ""
        echo "🎉 Integration Complete!"
        echo "======================"
        echo ""
        echo "🌐 To start the web interface:"
        echo "   docker run -p 5000:5000 pdf-extractor-web:latest"
        echo "   Then open: http://localhost:5000"
        echo ""
        echo "💻 To use command-line mode:"
        echo "   docker run --rm -v \$(pwd)/input:/app/input -v \$(pwd)/output:/app/output pdf-extractor-web:latest ./run.sh"
        echo ""
        echo "🧪 Testing the web server now..."
        
        # Start container in background for testing
        CONTAINER_ID=$(docker run -d -p 5000:5000 pdf-extractor-web:latest)
        
        # Wait a moment for startup
        sleep 3
        
        # Test health endpoint
        if curl -s http://localhost:5000/health >/dev/null 2>&1; then
            echo "✅ Web server is running and healthy!"
            echo "🌟 Open your browser to: http://localhost:5000"
            echo ""
            echo "Press any key to stop the test server..."
            read -n 1 -s
            docker stop $CONTAINER_ID >/dev/null 2>&1
            docker rm $CONTAINER_ID >/dev/null 2>&1
            echo "Test server stopped."
        else
            echo "⚠️  Web server started but health check failed"
            echo "   You can still try accessing: http://localhost:5000"
            docker stop $CONTAINER_ID >/dev/null 2>&1
            docker rm $CONTAINER_ID >/dev/null 2>&1
        fi
        
    else
        echo "❌ Docker build failed"
        echo "💡 Try these solutions:"
        echo "   1. Run: docker system prune -af"
        echo "   2. Use minimal requirements: cp requirements.minimal.txt requirements.txt"
        echo "   3. Check TROUBLESHOOTING.md"
    fi
else
    echo "❌ Docker not available"
    echo "💡 You can still run locally:"
    echo "   pip install -r requirements.txt"
    echo "   python web_server.py"
    echo "   Then open: http://localhost:5000"
fi

echo ""
echo "📚 Next Steps:"
echo "=============="
echo "1. 🌐 Web Mode: docker run -p 5000:5000 pdf-extractor-web:latest"
echo "2. 💻 CLI Mode: docker run --rm -v \$(pwd)/input:/app/input -v \$(pwd)/output:/app/output pdf-extractor-web:latest ./run.sh"
echo "3. 📖 Read WEB_UI_INTEGRATION.md for detailed instructions"
echo ""
echo "🏆 Your PDF extractor now has a beautiful web interface!"
echo "   Perfect for hackathon demos and testing! 🚀"