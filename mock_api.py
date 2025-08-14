from flask import Flask, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Define an API endpoint for getting order status
# The <order_id> part is a dynamic variable from the URL
@app.route('/get_order_status/<order_id>', methods=['GET'])
def get_order_status(order_id):
    """
    A mock function that simulates looking up an order status.
    In a real system, this would query a database.
    """
    print(f"API received request for order ID: {order_id}")
    
    # Mock data for different order IDs
    mock_database = {
        "12345": {"status": "Shipped", "delivery_date": "August 18th, 2025"},
        "67890": {"status": "Processing", "delivery_date": "August 20th, 2025"},
    }
    
    # Get the order info, or a default response if not found
    order_info = mock_database.get(order_id, {"status": "Not Found", "delivery_date": "Unknown"})
    
    # Return the data as a JSON response
    return jsonify(order_info)

# This makes the script runnable
if __name__ == '__main__':
    # Runs the app on localhost, port 5000
    app.run(debug=True, port=5000)
