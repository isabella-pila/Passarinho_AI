import os
import json
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .som import identificar


@csrf_exempt
def index(request):
    """Página principal."""
    resultado = None
    erro = None

    if request.method == "POST" and request.FILES.get("audio"):
        audio = request.FILES["audio"]

        # Salva arquivo temporário
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        caminho = os.path.join(settings.MEDIA_ROOT, audio.name)
        with open(caminho, "wb") as f:
            for chunk in audio.chunks():
                f.write(chunk)

        try:
            resultado = identificar(caminho, str(settings.DADOS_DIR))
            resultado["arquivo"] = audio.name
        except Exception as e:
            erro = str(e)
        finally:
            # Remove arquivo temporário
            if os.path.exists(caminho):
                os.remove(caminho)

    return render(request, "index.html", {
        "resultado": resultado,
        "resultado_json": json.dumps(resultado) if resultado else "null",
        "erro": erro,
    })
