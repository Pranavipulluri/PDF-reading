@echo off
echo 🧪 Testing PDF Extractor
echo ========================

REM Check if Docker image exists
docker images pdf-extractor:latest >nul 2>&1
if errorlevel 1 (
    echo ❌ PDF extractor not built yet!
    echo    Run 'instant_fix.bat' first to build the extractor.
    pause
    exit /b 1
)

REM Check if we have PDFs
if not exist "input\*.pdf" (
    echo ⚠️  No PDF files found in 'input' folder.
    echo    Please add PDF files to the 'input' folder first.
    pause
    exit /b 0
)

REM Count PDFs
for /f %%i in ('dir /b input\*.pdf 2^>nul ^| find /c /v ""') do set pdf_count=%%i
echo 📄 Found %pdf_count% PDF file(s) to process

REM Create output directory
if not exist "output" mkdir output

REM Run the extractor
echo 🚀 Running PDF extractor...
docker run --rm -v "%cd%\input:/app/input" -v "%cd%\output:/app/output" --network none pdf-extractor:latest

REM Check results
if exist "output\*.json" (
    echo.
    echo ✅ SUCCESS!
    echo 📊 Generated files:
    dir output\*.json
    echo.
    echo 📖 Sample output:
    for %%f in (output\*.json) do (
        echo --- %%f ---
        type "%%f"
        echo.
        goto :show_one
    )
    :show_one
) else (
    echo ❌ No output files were generated.
    echo    Check the logs above for any errors.
)

echo.
echo 🎯 Test complete!
pause