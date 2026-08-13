module {
  func.func @kernel(%value: i32) -> i32 {
    %result = arith.addi %value, %value : i32
    func.return %result : i32
  }
}
