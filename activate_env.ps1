# PowerShell script to activate nlpenv virtual environment
Write-Host "Activating nlpenv virtual environment..." -ForegroundColor Green
& ".\nlpenv\Scripts\Activate.ps1"
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the Speak2Data application:" -ForegroundColor Yellow
Write-Host "  streamlit run app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "To deactivate the virtual environment:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor Cyan
