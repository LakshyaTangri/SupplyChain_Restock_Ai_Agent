class RequestValidator:
    @staticmethod
    def validate_forecast_request(request: dict) -> bool:
        required_fields = ['product_id', 'horizon', 'features']
        return all(field in request for field in required_fields)

    @staticmethod
    def validate_inventory_request(request: dict) -> bool:
        required_fields = ['store_id', 'product_ids']
        return all(field in request for field in required_fields)

    @staticmethod
    def validate_optimization_request(request: dict) -> bool:
        required_fields = ['store_id', 'product_ids', 'optimization_type']
        return all(field in request for field in required_fields)

# Main server setup
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add services to the server
    add_DemandForecastServicer_to_server(DemandForecastService(), server)
    add_InventoryMonitorServicer_to_server(InventoryMonitorService(), server)
    
    # Start the server
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
