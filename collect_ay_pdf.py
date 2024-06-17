from Fundamentals import BSE


class FYReport:
    def __init__(self):
        self.bse = BSE()

    def download_fy_report(self, bse_code:str, start_fy: int = 2014, end_fy: int = 2023, dwn_path: str = 'Data/AnnualReports'):
        self.bse.download_annual_reports(bse_code, str(start_fy), str(end_fy), dwn_path)


# f = FYReport()
# f.download_fy_report('500033', start_fy=2014, end_fy=2015, dwn_path='.')