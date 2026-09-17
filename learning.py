from email import message
from fileinput import filename
import logging
from time import asctime
logging.basicConfig(level=logging.INFO,
format= "%(asctime)s %(levelname)s %(message)s",
filename = "Application.log"),

logging.debug("This is a debug message")
logging.info("This is an info message")
logging.error("This is an error message")
logging.warning("This is a warning mesage")
logging.critical("This is a critical message")

def divide(a : int, b:int)->float:
    logging.info("Divison of %s / %s", a, b)

    if b == 0:
        logging.error("Division of zero")
        return None

    result = a/b
    logging.info("Divisibilty completed result = %s", result)

    return result

divide(10,2)

#===================================================
import hashlib

text = "Prashant Kumar Singh"
hash_text = hashlib.sha256(text.encode()).hexdigest()
print(hash_text)

text = "Prashant Kumsr Singh AI learning is exciting"
def clean(text:str)->str:
   return "".join(ch if ch.isalnum() else "_" for ch in text)

print(clean(text))

from pathlib import Path

def dir_travel(file_path:Path)->str:

    print(file_path.name)

    return file_path.stem

file_path = Path("c:/Test/text/abc/health_report.txt")
file_stem = dir_travel(file_path)
print(file_stem)
    