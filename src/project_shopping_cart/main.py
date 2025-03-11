#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start
from crewai import LLM, Agent, Crew, Task, Process

from project_shopping_cart.crews.shopping_crew.shopping_crew import ShoppingCrew

from project_shopping_cart.crews.product_content_crew.product_content_crew import ProductContentCrew
import subprocess

from dotenv import load_dotenv,find_dotenv
_:bool = load_dotenv(find_dotenv())

gemini_llm = LLM(model='gemini/gemini-2.0-flash')


class ShoppingState(BaseModel):
    product_list: str = ""
    product_content:str = ""

class ShoppingFlow(Flow[ShoppingState]):

    @start()
    def research_product(self):
        print("Generating product list")

        try:
            result = (ShoppingCrew()
            .crew()
            .kickoff())
            print("Product List generated", result.raw)
            self.state.product_list = result.raw

        except Exception as e:
            print("Error generating product list", e)

    @listen(research_product)
    def generate_product(self):
        print("Generating Product Content")
        try:
            result = (
                ProductContentCrew()
                .crew()
                .kickoff(inputs={"product_list": self.state.product_list})
            )

            print("Product List generated", result.raw)
            self.state.product_content = result.raw
        except Exception as e:
            print("Error generating product content", e)

    @listen(generate_product)
    def save_content(self):
        return {
            "product_content": self.state.product_content
        }



def kickoff():
    shopping_flow = ShoppingFlow()
    shopping_flow.kickoff()


def plot():
    shopping_flow = ShoppingFlow()
    shopping_flow.plot()


def run_chainlit():
    subprocess.run(['chainlit', 'run', 'src/project_shopping_cart/app.py', '-w'])


if __name__ == "__main__":
    kickoff()
        