module {
  func.func @kernel(%value: i32) -> i32 {
    %result = arith.muli %value, %value : i32
    "linalg_vendor.marker"() : () -> ()
    func.return %result : i32
  }
}
