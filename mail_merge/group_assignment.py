
import sys
import re
import time

import smtplib

import pandas as pd

def main():
    """ 
    Reads in the student roster,
    iterates through each group,
    formats the message accordingly,
    and then sends the email to necessary recipients
    """
    
    # read in spreadsheet
    roster = pd.read_csv(sys.argv[1])
    roster.rename(columns={
        'Groups (Total: 15)': 'group', 
        'Student Name': 'name',
        'Email' : 'email'}, inplace=True)
    roster = roster[roster.group.notnull()]
    
    roster_names = roster.name.str.split(',', expand=True)
    roster_names.columns = ['last', 'first']
    
    # capture preferred names between '(' and ')'
    roster_names.loc[roster_names['first'].str.contains("\("), 'first'] = \
        roster_names[roster_names['first'].str.contains("\(")]['first'].str.split('(', expand=True)[1]
    roster_names['first'] = roster_names['first'].str.replace(')', '')
    
    # for everyone else, just get the first name in the column
    roster_names['first'] = roster_names['first'].str.lstrip().str.split(' ', expand=True)[0]

    # remove padding/spaces
    roster_names['first'] = roster_names['first'].str.rstrip().str.lstrip()
    roster_names['last'] = roster_names['last'].str.rstrip().str.lstrip()
    
    roster = pd.concat([roster, roster_names], axis=1)
    
    #set up the smtp server connection
    server = "smtp-server.its.caltech.edu"
    server = smtplib.SMTP_SSL(server, 465)
    #server.starttls()
    server.login('bebi103', 'CYN7GfkkdF')

    # set up email fields
    from_addr = "bebi103@caltech.edu"
    cc_addr = ["bois@caltech.edu"]
    bcc_addr = ["bebi103@caltech.edu"]
    subject = "BE/Bi 103: Group Assignment"
    
    
    # customize reciepients and message
    for thisgroup in roster.group.unique():
        roster_thisgroup = roster[roster.group == thisgroup]
        to_addr = list(roster_thisgroup.email)
        names_emails = str(roster_thisgroup[['first', 'last', 'email', 'group']])
        
        if thisgroup == 4 or len(roster_thisgroup) == 4:
            rearrange = "There may be reassignment as enrollment stabilizes."
        else:
            rearrange = ""
        
        roster_thisgroup['first']
        
        text = format_text(roster_thisgroup['first'], thisgroup, names_emails, rearrange)
        
        send_message(server, from_addr, to_addr, cc_addr, bcc_addr, subject, text)

    server.quit()

    return

def format_text(first_names, group_number, names_emails, rearrange):
    """ 
    Formats the body of the email message
    """
    first_names = list(first_names)
    first_names_str = ", ".join(first_names[:-1])
    first_names_str += ", and " + first_names[-1]
    
    text = """
Dear {first_names},

You are in group {group_number}. {special_msg}

{info}

Thanks,
BE/Bi 103 staff
    """.format(
        first_names = first_names_str, 
        group_number = int(group_number),
        info = names_emails,
        special_msg = rearrange)
        
    return text

def send_message(server, from_addr, to_addr, cc_addr, bcc_addr, subject, text):
    """ 
    Prepares and sends the actual message
    """

    message = """\
From: {from_addr}
To: {to_addr}
CC: {cc_addr}
Subject: {subject}

{text}
    """.format(
        from_addr = from_addr, 
        to_addr = ", ".join(to_addr), 
        cc_addr = ", ".join(cc_addr), 
        subject = subject, 
        text = text)

    # Send the mail            
    server.sendmail(from_addr, to_addr + cc_addr + bcc_addr, message)
    time.sleep(1)
    return
    
if __name__ == "__main__":
    main()

