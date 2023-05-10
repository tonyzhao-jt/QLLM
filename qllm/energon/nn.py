try:
    import colossalai as colossalai
    from colossalai.nn import Linear1D_Col, Linear1D_Row
    from colossalai.nn import VocabParallelEmbedding1D
except:
    print("colossalai module is not installed.")
    colossalai = None
    Linear1D_Col = None
    Linear1D_Row = None


# write low-precision quantization supported version of MultiHeadAttention1D
class QLinear1D_Col(Linear1D_Col):
    

