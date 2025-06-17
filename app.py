from backend.app import app

application = app

if __name__ == '__main__':
    application.run(debug=True, port=3000)
