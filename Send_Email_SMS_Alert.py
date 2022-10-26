import smtplib
import sys
from email.message import EmailMessage
import maskpass 
import os



def email_alert(subject,body,to):
    msg=EmailMessage()
    msg.set_content(body)
    msg['subject']=subject 
    msg['to']=to
    # USER ANME AND PASSWORD
    user="soumen.iitkgp.das@gmail.com"
    msg['from'] =user
    #passwd="xfdxwasxalaphpqn"
    print("\n")
    passwd=maskpass.askpass(prompt="Password:", mask="*")
    #SERVER ACCESS
    server  =smtplib.SMTP("smtp.gmail.com",587)
    server.starttls()
    server.login(user,passwd)
    server.send_message(msg)
    server.quit()
    pass

def whats_app():
    import pywhatkit
    pywhatkit.sendwhatmsg('+917407781433','Hello Soumen',16,45) # Ph no ,sms, hour ,minutes
    pass


def main():
    email_alert(sys.argv[1], sys.argv[2],sys.argv[3])
    print("\n","Subject =",sys.argv[1],"\n","Body=",sys.argv[2],"\n", "Email_Address_to=",sys.argv[3])
    whats_app()
    pass

main()