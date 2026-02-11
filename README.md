🚀 DÉPLOIEMENT SUR RENDER

Push sur GitHub

Connecter repo à Render

Choisir Web Service

Plan free

Déployer

Tu obtiens :

https://transformiq-api.onrender.com


Swagger auto :

https://transformiq-api.onrender.com/docs

À l’état actuel (endpoint /analyze avec UploadFile) tu as 3 façons simples de charger ton CSV.

✅ MÉTHODE 1 — Via Swagger (le plus simple)

Va sur :

https://ton-api.onrender.com/docs


Clique sur :

POST /analyze


Clique sur "Try it out"

Clique sur Choose File

Sélectionne ton test_data.csv

Clique sur Execute

👉 Résultat immédiat.

C’est la méthode la plus simple pour tester.

✅ MÉTHODE 2 — Via Postman

Ouvre Postman

Méthode : POST

URL :

https://ton-api.onrender.com/analyze


Body → form-data

Key = file

Type = File

Upload ton CSV

Send

✅ MÉTHODE 3 — Via cURL (terminal)
curl -X POST "https://ton-api.onrender.com/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_data.csv"
