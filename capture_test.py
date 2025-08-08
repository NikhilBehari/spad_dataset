import serial.tools.list_ports

def list_com_ports():
    """Lists all available COM ports and their descriptions."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No COM ports found.")
    else:
        print("Available COM ports:")
        for port in ports:
            print(f"  - Device: {port.device}")
            print(f"    Name: {port.name}")
            print(f"    Description: {port.description}")
            print(f"    Hardware ID: {port.hwid}")
            print("-" * 20)

if __name__ == "__main__":
    list_com_ports()