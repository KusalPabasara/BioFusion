"""
BioFusion Kiosk — Report Generator
Creates PDF reports and QR codes for mobile download.
"""

import os
import uuid
import qrcode
from datetime import datetime
from PIL import Image
import numpy as np
from io import BytesIO
import logging

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

# BioFusion palette
SAPPHIRE = HexColor("#2563EB")
EMERALD = HexColor("#10B981")
AMBER = HexColor("#F59E0B")
DARK = HexColor("#1E293B")
GRAY = HexColor("#64748B")


def generate_report_id():
    """Generate a unique report ID."""
    return str(uuid.uuid4())[:8]


def _pil_to_reportlab_image(pil_image, width_cm=8):
    """Convert a PIL Image to a ReportLab Image object."""
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    img = RLImage(buffer, width=width_cm * cm, height=width_cm * cm)
    return img


def _numpy_to_reportlab_image(np_array, width_cm=8):
    """Convert a numpy RGB array to a ReportLab Image object."""
    pil_image = Image.fromarray(np_array.astype(np.uint8))
    return _pil_to_reportlab_image(pil_image, width_cm)


def generate_qr_code(url, size=200):
    """
    Generate a QR code image for the given URL.

    Returns:
        PIL Image of the QR code
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1E293B", back_color="white")
    img = img.resize((size, size), Image.NEAREST)
    return img


def generate_pdf_report(report_id, capture_path, result, overlay, reports_dir):
    """
    Generate a PDF report with the X-ray analysis results.

    Args:
        report_id: unique report identifier
        capture_path: path to the captured X-ray image
        result: dict from inference.analyze_image()
        overlay: numpy RGB array of the Grad-CAM overlay
        reports_dir: directory to save the PDF

    Returns:
        pdf_path: path to the generated PDF
    """
    filename = f"report_{report_id}.pdf"
    pdf_path = os.path.join(reports_dir, filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'KioskTitle', parent=styles['Title'],
        fontSize=22, textColor=SAPPHIRE, spaceAfter=2 * mm,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        'KioskSubtitle', parent=styles['Normal'],
        fontSize=11, textColor=GRAY, alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )
    heading_style = ParagraphStyle(
        'KioskHeading', parent=styles['Heading2'],
        fontSize=14, textColor=DARK, spaceBefore=6 * mm, spaceAfter=3 * mm,
    )
    body_style = ParagraphStyle(
        'KioskBody', parent=styles['Normal'],
        fontSize=11, textColor=DARK, spaceAfter=2 * mm,
    )
    result_style = ParagraphStyle(
        'KioskResult', parent=styles['Normal'],
        fontSize=18, textColor=EMERALD if result["class_name"] == "NORMAL" else AMBER,
        alignment=TA_CENTER, spaceBefore=4 * mm, spaceAfter=4 * mm,
    )
    disclaimer_style = ParagraphStyle(
        'KioskDisclaimer', parent=styles['Normal'],
        fontSize=8, textColor=GRAY, alignment=TA_CENTER,
        spaceBefore=10 * mm,
    )

    elements = []

    # ── Header ──
    elements.append(Paragraph("🏥 BioFusion Kiosk", title_style))
    elements.append(Paragraph("AI-Assisted Pneumonia Detection Report", subtitle_style))

    # ── Report Info ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_data = [
        ["Report ID:", report_id],
        ["Date/Time:", timestamp],
        ["Model:", "ResNet50 (Fine-tuned on Pediatric Chest X-rays)"],
    ]
    info_table = Table(info_data, colWidths=[40 * mm, 120 * mm])
    info_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), GRAY),
        ('TEXTCOLOR', (1, 0), (1, -1), DARK),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 5 * mm))

    # ── Result ──
    elements.append(Paragraph("Analysis Result", heading_style))
    confidence_pct = f"{result['confidence'] * 100:.1f}%"
    result_text = f"<b>{result['class_name']}</b> — Confidence: {confidence_pct}"
    elements.append(Paragraph(result_text, result_style))

    # Probabilities
    prob_data = [
        ["Normal:", f"{result['probabilities']['normal'] * 100:.1f}%"],
        ["Pneumonia:", f"{result['probabilities']['pneumonia'] * 100:.1f}%"],
    ]
    prob_table = Table(prob_data, colWidths=[30 * mm, 30 * mm])
    prob_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGNMENT', (0, 0), (-1, -1), 'CENTER'),
        ('TEXTCOLOR', (0, 0), (-1, -1), DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, GRAY),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#F1F5F9")),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 5 * mm))

    # ── Images ──
    elements.append(Paragraph("Captured X-Ray &amp; Grad-CAM Analysis", heading_style))

    try:
        xray_img = _pil_to_reportlab_image(Image.open(capture_path), width_cm=7)
        overlay_img = _numpy_to_reportlab_image(overlay, width_cm=7)

        img_table = Table([[xray_img, overlay_img]], colWidths=[75 * mm, 75 * mm])
        img_table.setStyle(TableStyle([
            ('ALIGNMENT', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(img_table)

        # Captions
        cap_table = Table(
            [["Original X-Ray", "Grad-CAM Heatmap"]],
            colWidths=[75 * mm, 75 * mm]
        )
        cap_table.setStyle(TableStyle([
            ('ALIGNMENT', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), GRAY),
        ]))
        elements.append(cap_table)
    except Exception as e:
        elements.append(Paragraph(f"[Images unavailable: {e}]", body_style))

    # ── Disclaimer ──
    elements.append(Paragraph(
        "⚠️ <b>DISCLAIMER:</b> This is an AI-assisted screening tool and does NOT "
        "constitute a clinical diagnosis. Results must be reviewed and confirmed by a "
        "qualified radiologist or physician. Do not make treatment decisions based "
        "solely on this report.",
        disclaimer_style
    ))
    elements.append(Paragraph(
        "Powered by BioFusion — Team GMora | BioFusion Hackathon 2026",
        disclaimer_style
    ))

    # Build PDF
    doc.build(elements)
    logger.info(f"Generated PDF report: {pdf_path}")
    return pdf_path
