{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "HW_Python_Batch05.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOzKr3ySN68LchVP+lKMrjL",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Chainchigit/SQL/blob/main/Build%20Game%20and%20OOP.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**HW1 Game: rock_paper_scissors**\n"
      ],
      "metadata": {
        "id": "XtzUcCLtrnYA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import random"
      ],
      "metadata": {
        "id": "AM6Tq0_cb0gf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Game: rock_paper_scissors\n",
        "def rock_paper_scissors():\n",
        "    plays = (\"rock\",\"scissors\",\"paper\")\n",
        "    Win   = int(0)\n",
        "    Lose  = int(0) \n",
        "    Tie   = int(0)\n",
        "    Foul  = int(0)\n",
        "    \n",
        "    while True:\n",
        "        player = input(\"Select your Turn or exit : \")\n",
        "        Bot    = random.sample(plays, 1)[0]\n",
        "\n",
        "        if player == \"exit\":\n",
        "            print(\"End Game\")\n",
        "            print(\"Final Score\" \" | \",\"Win  : \",Win,\"Lose : \",Lose,\"Tie  : \",Tie, \"Foul : \",Foul) \n",
        "            break\n",
        "        if  player == Bot:\n",
        "            print(\"You Tie\")\n",
        "            Tie += 1\n",
        "        elif (player == plays[0] and Bot == plays[1] \n",
        "              or player == plays[1] and Bot == plays[2] \n",
        "              or player == plays[2] and Bot == plays[0]):\n",
        "            print(\"You WIN!!!\")\n",
        "            Win += 1\n",
        "        elif (player == plays[1] and Bot == plays[0] \n",
        "              or player == plays[2] and Bot == plays[1] \n",
        "              or player == plays[0] and Bot == plays[2]):\n",
        "            print(\"You Lose\") \n",
        "            Lose += 1     \n",
        "        else:\n",
        "            print(\"Incorrect Please try again.\")   \n",
        "            Foul += 1\n",
        "\n",
        "    "
      ],
      "metadata": {
        "id": "MSTV5yGrTi26"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rock_paper_scissors()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W8Ozo3f3cFIx",
        "outputId": "f0ea2897-9d9c-4e76-af0e-42aaaf65f402"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Select your Turn or exit : rock\n",
            "You Tie\n",
            "Select your Turn or exit : exit\n",
            "End Game\n",
            "Final Score |  Win  :  0 Lose :  0 Tie  :  1 Foul :  0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "HW2_OOP\n",
        "Object Oriented Programming\n",
        "class to create new Class **bold text**\n",
        "ATM methods\n",
        "\n"
      ],
      "metadata": {
        "id": "HBfzr_rqmS7p"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class ATM:\n",
        "    # constructor => initialization\n",
        "    def __init__(self, name, pin, account_balance, score_credits):\n",
        "        self.name = name\n",
        "        self.pin = pin \n",
        "        self.account_balance = account_balance\n",
        "        self.score_credits = score_credits\n",
        "       \n",
        "            \n",
        "    def __str__(self):\n",
        "        return f\"Hi there {self.name} welcome to ATM\"\n",
        "\n",
        "    # ATM methods (functions designed for a ATM)\n",
        "    # Method 1 Connect\n",
        "    def Login (self,pin):\n",
        "         if pin == self.pin:\n",
        "            print(\"Home page\")\n",
        "         else:\n",
        "            print(\"Don't have an account, please open an account.\") \n",
        "\n",
        "    # Method 2 Deposits\n",
        "    def deposit (self) :\n",
        "        amount = input(\"How much Tranfer in : \")\n",
        "        print(f\"Accout balance {self.account_balance + int(amount)}\")\n",
        "        self.account_balance = self.account_balance + int(amount)\n",
        "    \n",
        "    # Method 3 witdraw\n",
        "    def withdraw (self,withdraw)  :\n",
        "        if self.account_balance < withdraw:\n",
        "           print(\"Not's enough money in the account balance\")\n",
        "        else:\n",
        "           self.account_balance -= withdraw\n",
        "           print(f\"You tranfer out : {withdraw} Baht.\")\n",
        "           return self.account_balance\n",
        "   \n",
        "    # Method 4 Pay Bill\n",
        "    def pay (self,pay_bill):\n",
        "        How_pay = input(\"Pay Bill By Card? Yes or Not : \")\n",
        "        if How_pay == \"Yes\" :\n",
        "           print(\"used card\")\n",
        "           if self.score_credits:\n",
        "              self.account_balance -= pay_bill\n",
        "              print(f\"Your point {pay_bill * self.score_credits}\")                              \n",
        "        else:\n",
        "           print(\"Pay by other solution\")\n",
        "           print(f\"Your point {pay_bill * 0}\")\n",
        "\n",
        "    # Method 5 investment return\n",
        "    def investment_return (self):\n",
        "        if self.account_balance:\n",
        "            print(f\"Your Interest {self.account_balance * 0.05}\")\n",
        "            print(f\"Your account balance {self.account_balance + self.account_balance * 0.05}\")\n",
        "            \n",
        "           "
      ],
      "metadata": {
        "id": "tTu9o0LdrF2I"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# log in and sign in \n",
        "ATM1 = ATM(\"Henry\", 1234, 10000,0.03)\n",
        "ATM1.Login(1234)"
      ],
      "metadata": {
        "id": "QfdiKVfI-zmW",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "025b49ab-615f-4e3e-8d0c-6daf399d345c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Home page\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Deposit an account_balance\n",
        "ATM1.deposit()"
      ],
      "metadata": {
        "id": "Ktdczj3_nNj5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0747a633-6242-4084-8c22-03665ede3c7f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "How much Tranfer in : 1200\n",
            "Accout balance 11200\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Withdraw an account_balance\n",
        "ATM1.withdraw(2000)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jZ8Ad-T0CS4w",
        "outputId": "84680290-8c48-428c-c4c3-02d110ee7516"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "You tranfer out : 2000 Baht.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "9200"
            ]
          },
          "metadata": {},
          "execution_count": 182
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# checking account balance\n",
        "ATM1.pay(1200)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2bJE6H5dJ18P",
        "outputId": "0fb23ef1-fdd9-4ead-99dd-2fe383e464ed"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Pay Bill By Card? Yes or Not : Yes\n",
            "used card\n",
            "Your point 36.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Investment return \n",
        "ATM1.investment_return()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_GS_wI7eNYs_",
        "outputId": "e76b7677-5652-471b-b4e8-4159b3eb4237"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Your Interest 400.0\n",
            "Your account balance 8400.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(ATM1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3ZSMXAu0XN9U",
        "outputId": "efb960df-06c6-4d1a-e9ec-857c5aea51df"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Hi there Henry welcome to ATM\n"
          ]
        }
      ]
    }
  ]
}
