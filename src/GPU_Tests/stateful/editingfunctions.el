;;; Some Emacs Lisp functions to manipulate test output.

(defvar *batch-size* 10000)

(defun bandwidth (service-time)
  (floor (* (/ (/ service-time 1000000.0)) *batch-size*)))

(defun yank-bandwidth (service-time)
  (kill-new (number-to-string (bandwidth service-time))))

(defun yank-bandwidth-from-yanked-text ()
  (interactive)
  (if mark-active
      (progn
        (kill-ring-save (region-beginning) (region-end))
        (yank-bandwidth (string-to-number (car kill-ring))))
    (message "No region selected.")))
