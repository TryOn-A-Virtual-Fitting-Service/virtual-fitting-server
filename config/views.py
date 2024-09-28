from django.http import HttpResponse

def landing_page(request):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Virtual Fitting Server</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                color: #333;
            }
            .container {
                text-align: center;
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            h1 {
                color: #4CAF50;
            }
            p {
                font-size: 1.2rem;
            }
            .info {
                margin-top: 1rem;
                font-size: 1rem;
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to the Virtual Fitting Server</h1>
            <p>This is the landing page.</p>
            <p class="info">This server implements the StableVITON model for virtual fitting image generation.<br>
            Team info: CAU CSE - Sewon Min, Geon Lim, Sanghyun Na</p>
        </div>
    </body>
    </html>
    """
    return HttpResponse(html_content)