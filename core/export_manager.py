import os
import json
import time
import logging
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

class ExportManager:
    def __init__(self, exports_dir):
        self.exports_dir = exports_dir
        os.makedirs(self.exports_dir, exist_ok=True)

    def export(self, module_name, data_list, format_type, screenshot_path=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{module_name}_export_{timestamp}"
        
        if not data_list:
            return {"error": "No data to export"}

        filepath = ""
        try:
            if format_type == "CSV":
                filepath = os.path.join(self.exports_dir, f"{filename}.csv")
                df = pd.DataFrame(data_list)
                df.to_csv(filepath, index=False)
            
            elif format_type == "JSON":
                filepath = os.path.join(self.exports_dir, f"{filename}.json")
                with open(filepath, 'w') as f:
                    json.dump(data_list, f, indent=4)
                    
            elif format_type == "TXT":
                filepath = os.path.join(self.exports_dir, f"{filename}.txt")
                with open(filepath, 'w') as f:
                    f.write(f"--- {module_name} Report ---\n")
                    f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    for item in data_list:
                        for k, v in item.items():
                            f.write(f"{k}: {v}\n")
                        f.write("\n")
            
            elif format_type == "Excel (.xlsx)":
                filepath = os.path.join(self.exports_dir, f"{filename}.xlsx")
                df = pd.DataFrame(data_list)
                df.to_excel(filepath, index=False)
                
            elif format_type == "PDF Report":
                filepath = os.path.join(self.exports_dir, f"{filename}.pdf")
                c = canvas.Canvas(filepath, pagesize=letter)
                width, height = letter
                
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, height - 50, f"{module_name} Detection Report")
                
                c.setFont("Helvetica", 10)
                c.drawString(50, height - 70, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if screenshot_path and os.path.exists(screenshot_path):
                    # Try to draw the screenshot
                    try:
                        c.drawImage(screenshot_path, 50, height - 280, width=300, preserveAspectRatio=True)
                        y = height - 320
                    except Exception as e:
                        logger.error(f"Failed to embed screenshot: {e}")
                        y = height - 100
                else:
                    y = height - 100
                
                for item in data_list:
                    if y < 100:
                        c.showPage()
                        y = height - 50
                    
                    c.setFont("Helvetica-Bold", 10)
                    c.drawString(50, y, "- Entry -")
                    y -= 15
                    c.setFont("Helvetica", 10)
                    for k, v in item.items():
                        c.drawString(70, y, f"{k}: {v}")
                        y -= 15
                    y -= 10
                    
                c.save()
            else:
                return {"error": f"Unsupported format {format_type}"}

            logger.info(f"Export generated: {filepath}")
            return {
                "success": True,
                "filepath": filepath,
                "filename": os.path.basename(filepath)
            }
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {"error": str(e)}
