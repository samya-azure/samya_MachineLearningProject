
import requests
import json


wms_login_token_api_url = "http://192.168.2.26:73/WMSAPIPRODDEV/api/v1/UserLoginAuthenticationMaster/SecurityLoginServiceMobile"
wms_login_token_payload = {
    "CompanyCode": "WebDBConnection",
    "IsAcending": true,
    "PageRecordCount": 10,
    "PageSize": 1,
    "RequestData": {
        "Password": "Wms@1234",
        "usercode": "SysAdmin"
    },
    "SortColumn": "",
    "UserGuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "userId": 0
}


