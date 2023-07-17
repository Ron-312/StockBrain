from flask import Flask, request
import VDBfromCVS
import semanticQueriesVDB

app = Flask(__name__)

@app.route('/create_db', methods=['POST'])
def create_db():
    VDBfromCVS.create_vector_DB_from_csv()
    return {'message': 'Database created successfully'}

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json['query']
    result = semanticQueriesVDB.check_if_company_exist_in_tweet(query_text)
    return result

@app.route('/symbol', methods=['POST'])
def symbol():
    company_name = request.json['company_name']
    sentence = request.json['sentence']
    result = semanticQueriesVDB.symbol_per_company_name(company_name, sentence)
    return {'symbol': result}
    
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