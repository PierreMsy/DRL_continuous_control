class Config:
    
    def __init__(self, seed) -> None:
        self.seed = seed

    def __str__(self):

        representation = f"""
        seed : {self.seed}
        """
        return representation