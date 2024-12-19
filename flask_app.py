from flask import Flask, request, jsonify
from car_control import go_forward, go_backward, turn_right, turn_left
import time

app = Flask(__name__)
current_position = None
current_angle = None
distances = None
competitor_name = ""
start_time = None
start_time_threshold = 15

start_time = time.time()

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global competitor_name, start_time, current_position, current_angle 
    current_position = None
    current_angle = None
    data = request.json
    if 'competitor_name' in data:
        competitor_name = data['competitor_name']
        start_time = time.time()
    return jsonify({"status": "success"}),  200

@app.route('/check_competitor', methods=['GET'])
def check_competitor():
    global competitor_name, start_time, start_time_threshold, current_position, current_angle
    current_position = None
    current_angle = None
    competitor_name = competitor_name if time.time() - start_time < start_time_threshold else None        
    return jsonify({"competitor_name": competitor_name}), 200    

@app.route('/execute', methods=['POST'])
def execute_command():
    data = request.json
    command = data.get("command", {})
    action = command.get("action")
    
    if action == "forward":
        distance = command.get("distance", 0)
        go_forward(abs(distance)) if distance >= 0 else go_backward(abs(distance))
    elif action == "backward":
        distance = command.get("distance", 0)
        go_backward(abs(distance)) if distance >= 0 else go_forward(abs(distance))    
    elif action == "left":
        angle = command.get("angle", 0)
        turn_left(abs(angle)) if angle >= 0 else turn_right(abs(angle))
    elif action == "right":
        angle = command.get("angle", 0)
        turn_right(abs(angle)) if angle >= 0 else turn_left(abs(angle))
    else:
        return jsonify({"message": "Invalid action"}), 400

    return jsonify({"message": "Command executed"}), 200

@app.route('/update_pos', methods=['POST'])
def update_position():
    global current_position
    data = request.get_json()
    if 'position' in data:
        current_position = data['position']
        return jsonify({"status": "success", "position": current_position}), 200
    return jsonify({"status": "error", "message": "Missing position data"}), 400

@app.route('/update_ang', methods=['POST'])
def update_angle():
    global current_angle
    data = request.get_json()
    if 'angle' in data:
        current_angle = data['angle']
        return jsonify({"status": "success", "angle": current_angle}), 200
    return jsonify({"status": "error", "message": "Missing angle data"}), 400

@app.route('/get_pos', methods=['GET'])
def get_position():
    if current_position is not None:
        return jsonify({"position": current_position}), 200
    return jsonify({"status": "error", "message": "Position not available"}), 404

@app.route('/get_ang', methods=['GET'])
def get_angle():
    if current_angle is not None:
        return jsonify({"angle": current_angle}), 200
    return jsonify({"status": "error", "message": "Angle not available"}), 404

@app.route('/update_distances', methods=['POST'])
def update_distances():
    global distances
    distances = request.json.get("distances", [])
    return jsonify({"status": "success", "distances": distances}), 200

@app.route('/get_distances', methods=['GET'])
def get_distances():
    return jsonify({"distances": distances}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
