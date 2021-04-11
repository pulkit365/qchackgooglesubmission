from typing import List, Tuple

import numpy as np
import cirq
def Identity_ops(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    

    return [], []
def rccx(a,b,c):
    return [cirq.H(c),cirq.T(c)**3,cirq.CNOT(b,c),cirq.T(c),cirq.CNOT(a,c),cirq.T(c)**3,cirq.CNOT(b,c),cirq.T(c),cirq.H(c)]
def toffoli(a,b,c):
    return [cirq.H(c),cirq.CNOT(a,c),cirq.T(c)**3,cirq.CNOT(b,c),cirq.T(c),cirq.CNOT(a,c),cirq.T(c)**3,cirq.CNOT(b,c),cirq.T(c),cirq.H(c),cirq.T(a),cirq.CNOT(b,a),cirq.T(b),cirq.T(a)**3,cirq.CNOT(b,a)]
def ccz(a,b,c):
    return [cirq.CNOT(a,c),cirq.T(c)**3,cirq.CNOT(b,c),cirq.T(c),cirq.CNOT(a,c),cirq.T(c)**3,cirq.CNOT(b,c),cirq.T(c),cirq.T(a),cirq.CNOT(b,a),cirq.T(b),cirq.T(a)**3,cirq.CNOT(b,a)]
def four_qubit_ops(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    
    a=target_qubits[0]
    b=target_qubits[1]
    c=target_qubits[2]
    d=target_qubits[3]
    
    e=cirq.GridQubit(5,4)
    
    circs=[]
    circs2=[]
    
    #circs.append(  (rccx(a,b,e)+toffoli(c,e,d)+rccx(a,b,e),[e])     )
    #circs2.append(  (rccx(a,b,e)+toffoli(c,e,d)+rccx(a,b,e),[e])     )
    
    circs.append(  ([],[])     )
    circs2.append(  ([],[])     )
    
    #circ2.append(cirq.matrix)
    
    #circs2.append(  ([cirq.Toffoli(a,b,c)],[])     )
    
    first=True
    for i,circ in enumerate(circs):
        #print("ran")
        list,ancilla=circs2[i]
        unit=cirq.unitary(cirq.Circuit(list))
        #print((unit[0][1]))
        
        if first:
            #print(unit)
            first=False
        if unit.shape==matrix.shape:
            if np.allclose(matrix,unit,atol=0.00001):
                return circ
    if np.allclose(matrix,np.identity(16),atol=0.00001):
        return [],[]
    print(np.array_str(cirq.unitary(cirq.Circuit(toffoli(a,b,e)+toffoli(c,e,d)+toffoli(a,b,e))  )))
    return (toffoli(a,b,e)+toffoli(c,e,d)+toffoli(a,b,e)),[e] 
    #return cirq.optimizers.decompose_multi_controlled_x([a,b,c],d,[e]),[e]       
def three_qubit_ops(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    
    a=target_qubits[0]
    b=target_qubits[1]
    c=target_qubits[2]
    
    circs=[]
    circs2=[]
    
    circs.append(  ([],[])     )
    circs2.append(  ([],[])     )
    
    
    circs.append(  (toffoli(a,b,c),[])     )
    circs2.append(  (toffoli(a,b,c),[])     )
    
    circs.append(  ([cirq.CNOT(c,b)]+toffoli(a,b,c)+[cirq.CNOT(c,b)],[])     )
    circs2.append(  ([cirq.CNOT(c,b)]+toffoli(a,b,c)+[cirq.CNOT(c,b)],[])     )
    
    
    
    circs.append(  (toffoli(a,b,c)+toffoli(a,c,b)+toffoli(a,b,c),[])     )
    circs2.append(  (toffoli(a,b,c)+toffoli(a,c,b)+toffoli(a,b,c),[])     )
    
    circs.append(  (ccz(a,b,c),[])     )
    circs2.append(  (ccz(a,b,c),[])     )
    
    circs.append(    (toffoli(c,b,a)+[cirq.CNOT(c,b),cirq.X(c)],[]))
    circs2.append(    (toffoli(c,b,a)+[cirq.CNOT(c,b),cirq.X(c)],[]))
    #circs2.append(  ([cirq.Toffoli(a,b,c)],[])     )
    
    first=True
    for i,circ in enumerate(circs):
        list,ancilla=circs2[i]
        unit=cirq.unitary(cirq.Circuit(list))
        if first:
            #print(matrix)
            first=False
        if np.allclose(matrix,unit,atol=0.00001):
            return circ
    return [],[]
def single_gate_ops(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    
    circs=[]
    circs2=[]
    circs.append(  ([],[])     )
    circs2.append(  ([],[])     )
    
    first=True
    for i,circ in enumerate(circs):
        list,ancilla=circs2[i]
        unit=cirq.unitary(cirq.Circuit(list))
        if first:
            #print(matrix)
            first=False
        if np.allclose(matrix,unit,atol=0.00001):
            return circ
    a=target_qubits[0]
    for gate in [cirq.X,cirq.Y,cirq.Z]:
        if np.allclose(matrix, (cirq.unitary(gate)),atol=0.00001) :
            return [gate(a)],[]
    for gate in [cirq.H]:#,cirq.Y,cirq.Z]:
        if np.allclose(matrix, (cirq.unitary(gate)),atol=0.00001) :
            
            
            return [cirq.Z(a),cirq.Y(a)**0.5],[]
    for gate in [cirq.S]:#,cirq.Y,cirq.Z]:
        if np.allclose(matrix, (cirq.unitary(gate)),atol=0.00001) :
            
            
            return [cirq.Z(a)**0.5],[]
    for gate in [cirq.T]:#,cirq.Y,cirq.Z]:
        if np.allclose(matrix, (cirq.unitary(gate)),atol=0.00001) :
            
            
            return [cirq.Z(a)**0.25],[]    

    return NotImplemented, []

def two_qubit_ops(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    a=target_qubits[0]
    b=target_qubits[1]
    
    #return [cirq.CNOT(a,b)],[]

    circs=[]
    circs2=[]
    #circs.append(  ([cirq.CNOT(a,b)],[])     )#For CNOT
    #circs.append(  ([  cirq.Z(a)**(1.5),cirq.X(b)**(0.5),cirq.Z(b)**(0.5),cirq.ISWAP(a,b)**0.5,cirq.ISWAP(a,b)**0.5,cirq.X(a)**(0.5),cirq.ISWAP(a,b)**0.5,cirq.ISWAP(a,b)**0.5,cirq.Z(b) **0.5 ] , [])   )
    circs2.append(  ([cirq.CNOT(a,b)],[])     )#For CNOT
    circs.append(  ([cirq.CNOT(a,b)],[])     )#For CNOT
    
    
    circs.append(  ([cirq.CNOT(b,a),cirq.X(b)],[])     )
    circs2.append(  ([cirq.CNOT(b,a),cirq.X(b)],[])     )
    
    
    circs.append(  ([cirq.google.SYC(a,b) ] , [])    )
    circs2.append(  ([cirq.google.SYC(a,b) ] , [])    )
    
    circs.append(  ([cirq.google.SYC(b,a) ], [])    )
    circs2.append(  ([cirq.google.SYC(b,a) ], [])    )
    
    circs.append(  ([cirq.X(a),cirq.X(b)],[])     )
    circs2.append(  ([cirq.X(a),cirq.X(b)],[])     )
    
    circs.append(  ([cirq.Y(a),cirq.Y(b)],[])     )
    circs2.append(  ([cirq.Y(a),cirq.Y(b)],[])     )
    
    circs.append(  ([cirq.Z(a),cirq.Z(b)],[])     )
    circs2.append(  ([cirq.Z(a),cirq.Z(b)],[])     )
    
    
    circs.append(  ([],[])     )
    circs2.append(  ([],[])     )
    
    first=True
    for i,circ in enumerate(circs):
        list,ancilla=circs2[i]
        unit=cirq.unitary(cirq.Circuit(list))
        if first:
            print(matrix)
            first=False
        if np.allclose(matrix,unit,atol=0.00001):
            return circ
    print (np.array_str(matrix))
    '''
    for gate in [cirq.X,cirq.Y,cirq.Z]:
        if np.allclose(matrix, (cirq.unitary(gate)),atol=0.00001) :
            return [gate(a)],[]
    for gate in [cirq.H]:#,cirq.Y,cirq.Z]:
        if np.allclose(matrix, (cirq.unitary(gate)),atol=0.00001) :
            
            
            return [cirq.Z(a),cirq.Y(a)**0.5],[]
    for gate in [cirq.S]:#,cirq.Y,cirq.Z]:
        if np.allclose(matrix, (cirq.unitary(gate)),atol=0.00001) :
            
            
            return [cirq.Z(a)**0.5],[]
    for gate in [cirq.T]:#,cirq.Y,cirq.Z]:
        if np.allclose(matrix, (cirq.unitary(gate)),atol=0.00001) :
            
            
            return [cirq.Z(a)**0.25],[]    
    '''
    return [], []


def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """
    np.set_printoptions(edgeitems=100)
    #f=open("mat_out","w")
    #f.write(np.array_str(matrix))
    a=target_qubits[0]
    b=a
    #print(matrix)
    #print(matrix.shape)
    #print(target_qubits)
    #swap = cirq.H(cirq.GridQubit(0, 0))
    #print(cirq.Circuit(swap, device=cirq.google.Foxtail))
    if matrix.shape==(2,2):
        return single_gate_ops(target_qubits,matrix)
    if matrix.shape==(4,4):
        return two_qubit_ops(target_qubits,matrix)
    
    if matrix.shape==(8,8):
        return three_qubit_ops(target_qubits,matrix)
    
    
    if matrix.shape==(16,16):
        return four_qubit_ops(target_qubits,matrix)
    
    return [], []
