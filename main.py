from flask import Flask, request
from VDBfromCVSClass import CompanyDBManager 
import semanticQueriesVDB
import os

app = Flask(__name__)

# Initialize your DB Manager with the CSV file path
db_manager = CompanyDBManager('./Data/nasdaq_symbols_to_name3_test.csv')

@app.route('/create_db', methods=['POST'])
def create_db():
    try:
        db_manager.create_vector_DB_from_csv()
    except Exception as e:
        return {'message': str(e)}, 500
    return {'message': 'Database created successfully'}, 200

@app.route('/update_db', methods=['POST'])
def update_db():
    try:
        # Assuming the new entries are sent in the request's JSON body
        new_entries_json_path = "./Data/nasdaq_full_tickers_test.json"
        db_manager.run_update_and_create_vector_db(new_entries_json_path)
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    return jsonify({'message': 'Database updated successfully'}), 200

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query')
    if not query_text:
        return {'message': 'query parameter is missing'}, 400
    try:
        result = semanticQueriesVDB.check_if_company_exist_in_tweet(query_text)
    except Exception as e:
        return {'message': str(e)}, 500
    return result, 200

@app.route('/symbol', methods=['POST'])
def symbol():
    company_name = request.json.get('company_name')
    sentence = request.json.get('sentence')
    if not company_name or not sentence:
        return {'message': 'company_name or sentence parameter is missing'}, 400
    try:
        result = semanticQueriesVDB.symbol_per_company_name(company_name, sentence)
    except Exception as e:
        return {'message': str(e)}, 500
    return {'symbol': result}, 200

@app.route('/shutdown', methods=['POST'])
def shutdown():
    print('Shutting down gracefully...')
    os._exit(0)  # exits the process immediately

    
if __name__ == '__main__':
    print("""
  ______    __                          __              _______                      __           
 /      \  /  |                        /  |            /       \                    /  |          
/$$$$$$  |_$$ |_     ______    _______ $$ |   __       $$$$$$$  |  ______   ______  $$/  _______  
$$ \__$$// $$   |   /      \  /       |$$ |  /  |      $$ |__$$ | /      \ /      \ /  |/       \ 
$$      \$$$$$$/   /$$$$$$  |/$$$$$$$/ $$ |_/$$/       $$    $$< /$$$$$$  |$$$$$$  |$$ |$$$$$$$  |
 $$$$$$  | $$ | __ $$ |  $$ |$$ |      $$   $$<        $$$$$$$  |$$ |  $$/ /    $$ |$$ |$$ |  $$ |
/  \__$$ | $$ |/  |$$ \__$$ |$$ \_____ $$$$$$  \       $$ |__$$ |$$ |     /$$$$$$$ |$$ |$$ |  $$ |
$$    $$/  $$  $$/ $$    $$/ $$       |$$ | $$  |      $$    $$/ $$ |     $$    $$ |$$ |$$ |  $$ |
 $$$$$$/    $$$$/   $$$$$$/   $$$$$$$/ $$/   $$/       $$$$$$$/  $$/       $$$$$$$/ $$/ $$/   $$/ 
    """)
    app.run(port=5000)
    print("Flask server has started successfully on port 5000")
