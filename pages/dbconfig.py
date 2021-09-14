host = '36fbc881-0ee0-4-231-b9ee.mongo.cosmos.azure.com'
port = '10255'
username = '36fbc881-0ee0-4-231-b9ee'
password = 'YmAp45GrVJUS9uJYthNrf0dhcZXs0GhuvXCeCemp1DT1Gjy5mm3zIkt0ZDOpLOWQwipBTmzwuuTqcY89xtXfSA=='
db_name = 'street-analytics'
video_collection = 'video-analytics'
image_collection = ''
args = 'ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@36fbc881-0ee0-4-231-b9ee@&ssl_cert_reqs=CERT_NONE'

# Connection URI
connection_string = 'mongodb://' + username + ':' + password + '@' + host + ':' + port + '/' + db_name + '?' + args