
def send_run_email(run_time:str):
	"""Function for sending an email.  Inputs the model runtime into the
	docstrings via decorator for easy formatting of the HTML body of an email.

	Args:
		url (str): [url of the listing]

	Returns:
		[None]: [Just sends the email.  Doesn't return anything]
	"""	
	import smtplib, ssl
	from email.mime.text import MIMEText
	from email.mime.multipart import MIMEMultipart

	def inputdata(run_time:str):
		"""Formats the runtime into an HTML format

		Args:
			run_time (str): Time it took to run the function

		Returns:
			html (str): HTML formatted response with run time
		"""	
		html="""
			<html>
				<body>
					<p>Your ECG is done!<br>
					Your ECG was processed and """ +str(run_time)+ """<br>
					Thank you!
					</p>
				</body>
			</html>
			"""
		return html

	with open('./src/rad_ecg/secret/login.txt') as login_file:
		login = login_file.read().splitlines()
		sender_email = login[0].split(':')[1]
		password = login[1].split(':')[1]
		receiver_email = login[2].split(':')[1]
		
	# Establish a secure session with gmail's outgoing SMTP server using your gmail account
	smtp_server = "smtp.gmail.com"
	port = 465

	message = MIMEMultipart("alternative")
	message["Subject"] = "Model is Finished!"
	message["From"] = sender_email
	message["To"] = receiver_email

	html = inputdata(run_time)

	attachment = MIMEText(html, "html")
	message.attach(attachment)
	context = ssl.create_default_context()

	with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
		server.login(sender_email, password)		
		server.sendmail(sender_email, receiver_email, message.as_string())

