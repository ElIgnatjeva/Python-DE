from datetime import datetime
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import numpy as np


class Transaction:
    def __init__(
        self, type: str, sum: float, time: datetime, balance: float, status: str
    ):
        self.t_type = type
        self.t_sum = sum
        self.t_time = time
        self.t_balance = balance
        self.t_status = status

    def __str__(self) -> str:
        return (
            f"Операция с типом {self.t_type} на сумму {self.t_sum} "
            f"в {self.t_time.strftime('%Y-%m-%d %H:%M')}, "
            f"остаток на счету {self.t_balance}, статус {self.t_status})"
        )

    def __repr__(self) -> str:
        return (
            f"Transaction(t_type={self.t_type}, t_sum={self.t_sum}, "
            f"t_time={self.t_time.strftime('%Y-%m-%d %H:%M')}, "
            f"t_balance={self.t_balance}, t_status={self.t_status})"
        )

    def to_dict(self) -> dict:
        return {
            "Type": self.t_type,
            "Sum": self.t_sum,
            "Time": self.t_time,
            "Balance": self.t_balance,
            "Status": self.t_status,
        }


class Account:
    _account_counter: int = 100000

    def __init__(self, account_holder: str, account_type: str, balance: float = 0):
        if balance < 0:
            raise ValueError("balance can't be negative")

        pattern = r"^[A-Z]+ [A-Z]+$"
        if not re.match(pattern, account_holder):
            raise ValueError("Invalid account holder name")

        Account._account_counter += 1
        self.account_holder = account_holder
        self.account_counter = Account._account_counter
        self._balance = balance
        self.operations_history: List[Transaction] = []
        self.account_number = f"ACC-{self.account_counter}"
        self._account_type = account_type

    def _add_transaction(self, type, sum, time, balance, status):
        new_transaction: Transaction = Transaction(type, sum, time, balance, status)
        self.operations_history.append(new_transaction)

    def deposit(self, amount: float):
        if amount < 1:
            self._add_transaction(
                "deposit", amount, datetime.now(), self.get_balance(), "fail"
            )
            raise ValueError("Amount of deposit must be positive")
        self._balance += amount
        self._add_transaction(
            "deposit", amount, datetime.now(), self.get_balance(), "success"
        )

    def withdraw(self, amount: float):
        if amount < 1:
            self._add_transaction(
                "withdraw", amount, datetime.now(), self.get_balance(), "fail"
            )
            print("Amount of withdrawal must be positive")
            return

        if self._balance < amount:
            self._add_transaction(
                "withdraw", amount, datetime.now(), self.get_balance(), "fail"
            )
            print("Amount of withdrawal exceeds user balance")
            return

        self._balance -= amount
        self._add_transaction(
            "withdraw", amount, datetime.now(), self.get_balance(), "success"
        )

    def get_balance(self) -> float:
        return self._balance

    def get_history(self) -> List[Transaction]:
        return self.operations_history.copy()

    def plot_history(self):
        transactions: List[Transaction] = self.get_history()
        transaction_list = [t.to_dict() for t in transactions]
        df = pd.DataFrame(transaction_list)

        plt.plot(df["Time"], df["Balance"])
        plt.xlabel("Time")
        plt.ylabel("Balance")
        plt.title("Динамика изменения баланса")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y\n%H:%M:%S"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        plt.show()

    def analyze_transactions(self, n: int):
        transactions: List[Transaction] = self.get_history()
        transaction_list = [t.to_dict() for t in transactions]
        print(transaction_list)
        df = pd.DataFrame(transaction_list)
        return df.sort_values(by="Sum", ascending=False).head(n)

    def clean_history(self, filename: str):
        df = pd.read_csv(filename)

        df["date_parsed"] = df["date"].apply(self._parse_date)
        df_valid_dates = df[df["date_parsed"].notna()].copy()

        valid_operations = {
            "checking": ["deposit", "withdraw"],
            "savings": ["deposit", "withdraw", "interest"],
        }

        df_valid_dates["is_valid"] = (
            (df_valid_dates["account_type"] == self._account_type)
            & df_valid_dates["operation"].isin(
                valid_operations.get(self._account_type, [])
            )
            & (df_valid_dates["status"] == "success")
            & (
                (
                    (df_valid_dates["operation"].isin(["deposit", "withdraw"]))
                    & df_valid_dates["amount"].apply(self._is_valid_amount)
                )
                | (df_valid_dates["operation"] == "interest")
            )
        )

        result_df = df_valid_dates[df_valid_dates["is_valid"]].drop("is_valid", axis=1)
        return result_df.sort_values(by="date_parsed").drop("date_parsed", axis=1)

    def _parse_date(self, date_str: str):
        date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y",
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        return None

    def _is_valid_amount(self, amount: float):
        try:
            amount_float = float(amount)
            return (
                amount_float > 0
                and pd.notna(amount_float)
                and np.isfinite(amount_float)
            )
        except (ValueError, TypeError):
            return False

    def load_transactions(self, filename: str):
        df = self.clean_history(filename)
        account_transactions = df[df["account_number"] == self.account_number]

        if len(self.get_history()) != 0:
            self._balance = (
                account_transactions.iloc[0]["balance_after"]
                - account_transactions.iloc[0]["amount"]
            )

        for index, at in account_transactions.iterrows():
            transaction_date = self._parse_date(at["date"])
            self._add_transaction(
                at["operation"],
                at["amount"],
                transaction_date,
                at["balance_after"],
                at["status"],
            )
            self._balance = at["balance_after"]


class CheckingAccount(Account):
    def __init__(self, account_type: str, account_holder: str, balance: float = 0):
        super().__init__(account_holder, account_type, balance)


class SavingsAccount(Account):
    def __init__(self, account_type: str, account_holder: str, balance: float = 0):
        super().__init__(account_holder, account_type, balance)

    def withdraw(self, amount: float):
        if amount < 1:
            self._add_transaction(
                "withdraw", amount, datetime.now(), self.get_balance(), "fail"
            )
            print("Amount of withdrawal must be positive")
            return

        if amount > self._balance / 2:
            self._add_transaction(
                "withdraw", amount, datetime.now(), self.get_balance(), "fail"
            )
            print("Withdrawal of more than half of the current balance is forbidden")
            return

        self._balance -= amount
        self._add_transaction(
            "withdraw", amount, datetime.now(), self.get_balance(), "success"
        )

    def apply_interest(self, rate: float):
        amount = self._balance * (rate / 100)
        self._balance += amount
        self._add_transaction(
            "interest", amount, datetime.now(), self.get_balance(), "success"
        )


# ======================================== TESTS ========================================


def test_account_creation():
    """Тестирование создания счетов и валидации"""
    print("\n=== Тест 1: Создание счетов ===")

    # Успешное создание
    acc1 = CheckingAccount("checking", "IVAN IVANOV", 1000)
    acc2 = SavingsAccount("savings", "ALICE BROWN", 2000)

    print(
        f"✅ Создан CheckingAccount: {acc1.account_number}, баланс: {acc1.get_balance()}"
    )
    print(
        f"✅ Создан SavingsAccount: {acc2.account_number}, баланс: {acc2.get_balance()}"
    )

    # Тест валидации имени
    try:
        Account("invalid name", "checking", 100)
        print("❌ Не сработала валидация имени")
    except ValueError as e:
        print(f"✅ Валидация имени работает: {e}")

        # Тест валидации баланса
    try:
        Account("IVAN IVANOV", "checking", -100)
        print("❌ Не сработала валидация баланса")
    except ValueError as e:
        print(f"✅ Валидация баланса работает: {e}")


def test_checking_account_operations():
    """Тестирование операций CheckingAccount"""
    print("\n=== Тест 2: Операции CheckingAccount ===")

    acc = CheckingAccount("checking", "JOHN DOE", 1000)

    # Успешный депозит
    acc.deposit(500)
    assert acc.get_balance() == 1500, "Ошибка депозита"
    print(f"✅ Депозит: баланс {acc.get_balance()}")

    # Успешное снятие
    acc.withdraw(300)
    assert acc.get_balance() == 1200, "Ошибка снятия"
    print(f"✅ Снятие: баланс {acc.get_balance()}")

    # Неудачное снятие (недостаточно средств)
    initial_balance = acc.get_balance()
    acc.withdraw(5000)
    assert acc.get_balance() == initial_balance, "Баланс изменился при неудачном снятии"
    print("✅ Проверка недостатка средств работает")

    # Неудачный депозит (отрицательная сумма)
    try:
        acc.deposit(-100)
        print("❌ Не сработала валидация отрицательного депозита")
    except ValueError:
        print("✅ Валидация отрицательного депозита работает")

    print(f"✅ Финальный баланс: {acc.get_balance()}")
    print(f"✅ История операций: {len(acc.get_history())} записей")


def test_savings_account_operations():
    """Тестирование операций SavingsAccount с ограничениями"""
    print("\n=== Тест 3: Операции SavingsAccount ===")

    acc = SavingsAccount("savings", "JANE SMITH", 1000)

    # Успешное снятие (менее 50%)
    acc.withdraw(400)
    assert acc.get_balance() == 600, "Ошибка снятия"
    print(f"✅ Снятие до 50%: баланс {acc.get_balance()}")

    # Неудачное снятие (более 50%)
    initial_balance = acc.get_balance()
    acc.withdraw(400)  # 400 > 600/2 = 300
    assert acc.get_balance() == initial_balance, (
        "Баланс изменился при превышении лимита"
    )
    print("✅ Ограничение на снятие более 50% работает")

    # Начисление процентов
    acc.apply_interest(10)  # 10% от 600 = 60
    assert acc.get_balance() == 660, "Ошибка начисления процентов"
    print(f"✅ Начисление процентов: баланс {acc.get_balance()}")

    print(f"✅ Финальный баланс: {acc.get_balance()}")


def test_load_transactions():
    """Тестирование загрузки транзакций из CSV"""
    print("\n=== Тест 4: Загрузка транзакций из CSV ===")

    # Создаем счет с номером, который есть в тестовом CSV
    Account._account_counter = 100000  # Сбрасываем счетчик

    acc = CheckingAccount("checking", "TEST USER", 0)
    acc.account_number = "ACC-100001"

    print(f"Тестируем счет: {acc.account_number}")

    # Загружаем транзакции
    initial_history_count = len(acc.get_history())
    acc.load_transactions("HW5/transactions_dirty.csv")
    final_history_count = len(acc.get_history())

    print(f"✅ Загружено транзакций: {final_history_count - initial_history_count}")
    print(f"✅ Финальный баланс: {acc.get_balance()}")

    # Проверяем, что загрузились только валидные транзакции
    valid_transactions = [t for t in acc.get_history() if t.t_status == "success"]
    assert len(valid_transactions) > 0, "Не загружено валидных транзакций"

    # Проверяем сортировку по дате
    dates = [t.t_time for t in acc.get_history()]
    assert dates == sorted(dates), "Транзакции не отсортированы по дате"
    print("✅ Транзакции отсортированы по дате")

    # Проверяем, что невалидные транзакции отфильтрованы
    invalid_operations = [t for t in acc.get_history() if t.t_sum <= 0]
    assert len(invalid_operations) == 0, "Загружены невалидные транзакции"
    print("✅ Невалидные транзакции отфильтрованы")


def test_analyze_transactions():
    """Тестирование анализа транзакций"""
    print("\n=== Тест 5: Анализ транзакций ===")

    acc = CheckingAccount("checking", "ANALYSIS TEST", 1000)
    acc.deposit(500)
    acc.withdraw(200)
    acc.deposit(1000)
    acc.withdraw(100)

    # Анализ самых крупных транзакций
    top_transactions = acc.analyze_transactions(2)
    print("✅ Топ-2 транзакции:")
    print(top_transactions[["Type", "Sum"]].to_string(index=False))

    # Проверяем, что транзакции отсортированы по убыванию суммы
    sums = top_transactions["Sum"].tolist()
    assert sums == sorted(sums, reverse=True), "Транзакции не отсортированы по сумме"
    print("✅ Транзакции правильно отсортированы по сумме")


def test_clean_history():
    """Тестирование очистки истории"""
    print("\n=== Тест 6: Очистка истории ===")

    acc = CheckingAccount("checking", "CLEAN TEST", 0)
    acc.account_number = "ACC-100001"  # Используем существующий в CSV

    cleaned_df = acc.clean_history("HW5/transactions_dirty.csv")

    print(f"✅ Валидных транзакций в CSV: {len(cleaned_df)}")
    print("✅ Столбцы результата:", list(cleaned_df.columns))

    # Проверяем, что остались только checking операции
    assert all(cleaned_df["account_type"] == "checking"), (
        "Есть транзакции других типов счетов"
    )

    # Проверяем, что остались только успешные операции
    assert all(cleaned_df["status"] == "success"), "Есть неуспешные транзакции"

    print("✅ Все проверки очистки истории пройдены")


def test_error_handling():
    """Тестирование обработки ошибок"""
    print("\n=== Тест 7: Обработка ошибок ===")

    acc = CheckingAccount("checking", "ERROR TEST", 100)

    # Тест загрузки из несуществующего файла
    try:
        acc.load_transactions("nonexistent.csv")
        print("❌ Не сработала обработка несуществующего файла")
    except Exception as e:
        print(f"✅ Обработка несуществующего файла: {type(e).__name__}")

    # Тест с CSV с неправильной структурой
    try:
        with open("invalid.csv", "w") as f:
            f.write("invalid,data\n1,2,3")
        acc.load_transactions("invalid.csv")
        print("✅ Система устойчива к неправильным CSV")
    except Exception as e:
        print(f"✅ Обработка неправильного CSV: {type(e).__name__}")


if __name__ == "__main__":
    test_account_creation()
    test_checking_account_operations()
    test_savings_account_operations()
    test_load_transactions()
    test_analyze_transactions()
    test_clean_history()
    test_error_handling()

    # ====== Дополнительная проверка загрузки из csv с динамическим присвоением account number ======

    # checking_account = CheckingAccount("checking", "IVAN IVANOV")
    # checking_account.load_transactions("HW5/transactions_dirty.csv")
    # print(checking_account.get_balance())
    # # checking_account.plot_history()

    # savings_account = SavingsAccount("savings", "IVAN IVANOV")
    # savings_account.load_transactions("HW5/transactions_dirty.csv")
    # print(savings_account.get_balance())
    # # savings_account.plot_history()

    # checking_account = CheckingAccount("checking", "IVAN IVANOV")
    # checking_account.load_transactions("HW5/transactions_dirty.csv")
    # print(checking_account.get_balance())
    # # checking_account.plot_history()

    # savings_account = SavingsAccount("savings", "IVAN IVANOV")
    # savings_account.load_transactions("HW5/transactions_dirty.csv")
    # print(savings_account.get_balance())
    # print(savings_account.analyze_transactions(6))
    # savings_account.plot_history()
