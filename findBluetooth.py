from bluepy.btle import Scanner, DefaultDelegate
import bluetooth
import threading

class ScanDelegate(DefaultDelegate):
    def __init__(self):
        DefaultDelegate.__init__(self)
        
    def handleDiscovery(self, dev, isNewDev, isNewData):
        if isNewDev:
            print("Discovered Beacon device", dev.addr, "with RSSI:",dev.rssi," dB")
        elif isNewData:
            print ("Received new data from Beacon device addr:", dev.addr)
            
def scanBeacon():
    scannerbyDevices = bluetooth.discover_devices(lookup_names = True)
    for addr, name in scannerbyDevices:
        print("Discovered Bluetooth device", addr, "with name:",name)



scanBeaconThread = threading.Thread(target = scanBeacon)
scanBeaconThread.start()

scanner = Scanner().withDelegate(ScanDelegate())
devices = scanner.scan(10.0)

scanBeaconThread.join()
print("Done")