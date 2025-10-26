from datetime import datetime

class Transaction:
    def __init__(self, type, sum, time, balance, status):
        self.t_type = type
        self.t_sum = sum
        self.t_time = time
        self.t_balance = balance
        self.t_status = status

class Account:
    _account_counter: int = 1000
    holder: str
    account_number: str
    _balance: float
    operations_history: Transaction = []

    def __init__(self, account_holder, balance=0):
        self.account_holder = account_holder
        self.balance = balance
        # todo: хранить счета пользователей, назначать порядковый номер
        account_number = "ACC-0001"

    def deposit(self, amount):
        if amount < 1:
            raise ValueError("Amount of deposit must be positive")
        self.balance = self.balance + amount
        new_transaction: Transaction = Transaction("deposit", amount, datetime.now(), self.balance, "success")
        self.operations_history.append(new_transaction)
        
    def withdraw(self, amount) :
        if amount < 1:
            raise ValueError("Amount of withdrawal must be positive")
        
        if self._balance < amount:
            raise ValueError("Amount of withdrawal exceeds user balance")
        
        # проверяет, достаточно ли средств на счёте, если нет — операция
        # не проходит, но ее попытка с статусом 'fail' все равно фиксируется
        # в истории;
        # try:
        #     self._balance -= amount
        # except:

    def get_balance(self): 
        return self.balance
    
    def get_history(self):
        return self.operations_history
    
class CheckingAccount(Account):
    def __init__(self, account_type, account_holder, balance=0):
        super().__init__(account_holder, balance)
        self._account_type = account_type

    def withdraw(self, amount):
        if amount > self._balance / 2:
            raise ValueError("Withdrawal of more than half of the current balance is forbidden")

class SavingsAccount(Account):
    def __init__(self, account_type, account_holder, balance=0):
        super().__init__(account_holder, balance)
        self._account_type = account_type

    def apply_interest(self, rate):
        self._balance * (rate / 100)

def main():
    acc : Account = Account("Elena I", 0)
    acc.deposit(10000)
    acc.get_balance()
