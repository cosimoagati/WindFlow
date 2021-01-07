(asdf:load-system :fare-csv)
(ql:quickload :vgplot)

(defun get-points (filename)
  (let ((file (fare-csv:read-csv-file filename)))
    (mapcar (lambda (list) (mapcar #'parse-integer list))
            (mapcar #'rest (apply #'mapcar #'list file)))))

(defun plot-points (filename index)
  (let ((points (get-points filename)))
    (vgplot:plot (first points) (elt points index))))
