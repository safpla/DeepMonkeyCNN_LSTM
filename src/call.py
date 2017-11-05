class Call():
    """
    properties:
        subject
        session_name
        channel_num
        begin_time
        end_time
        call_type
        sig: optional
    """

    def __init__(self, subject, session_num, channel_num, begin_time, end_time, call_type):
        self.subject = subject
        self.session_num = session_num
        self.channel_num = channel_num
        self.begin_time = begin_time
        self.end_time = end_time
        self.call_type = call_type
