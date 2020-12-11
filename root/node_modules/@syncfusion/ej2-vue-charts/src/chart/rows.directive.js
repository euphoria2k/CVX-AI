var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    }
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import Vue from 'vue';
import { EJComponentDecorator } from '@syncfusion/ej2-vue-base';
var RowsDirective = /** @class */ (function (_super) {
    __extends(RowsDirective, _super);
    function RowsDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    RowsDirective.prototype.render = function () {
        return;
    };
    RowsDirective = __decorate([
        EJComponentDecorator({})
    ], RowsDirective);
    return RowsDirective;
}(Vue));
export { RowsDirective };
export var RowsPlugin = {
    name: 'e-rows',
    install: function (Vue) {
        Vue.component(RowsPlugin.name, RowsDirective);
    }
};
var RowDirective = /** @class */ (function (_super) {
    __extends(RowDirective, _super);
    function RowDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    RowDirective.prototype.render = function () {
        return;
    };
    RowDirective = __decorate([
        EJComponentDecorator({})
    ], RowDirective);
    return RowDirective;
}(Vue));
export { RowDirective };
export var RowPlugin = {
    name: 'e-row',
    install: function (Vue) {
        Vue.component(RowPlugin.name, RowDirective);
    }
};
